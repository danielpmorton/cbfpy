"""
# CBF Configuration class

## Defining the problem:

CBFs have two primary implementation requirements: the dynamics functions, and the barrier function(s).
These can be specified through the `f`, `g`, and `h` methods, respectively. Note that the main requirements
for these functions are that (1) the dynamics are control-affine, and (2) the barrier function(s) are "zeroing"
barriers, as opposed to "reciprocal" barriers. A zeroing barrier is one which is positive in the interior of the
safe set, and zero on the boundary.

Depending on the relative degree of your barrier function(s), you should implement the `h_1` method
(for a relative-degree-1 barrier), and/or the `h_2` method (for a relative-degree-2 barrier).

## Tuning the CBF:

The CBF config provides a default implementation of the CBF "gain" function `alpha`, and `alpha_2` for
relative-degree-2 barriers. To change the sensitivity of the CBF, these functions can be modified to
increase or decrease the effect of the barrier(s). For instance, `alpha(h) = h` is the default implementation,
but to increase the sensitivity of the CBF, one could use `alpha(h) = 2 * h`. The only requirements for these
functions are that they are monotonically increasing and pass through the origin (class Kappa functions).

The CBFConfig also provides a default implementation of the CBF QP objective function, which is to minimize
the norm of the difference between the safe control input and the desired control input. This can also be modified
through the `P` and `q` methods, which define the quadratic and linear terms in the QP objective, respectively. This
does require that P is positive semi-definite.

## Relaxation:

Depending on the construction of the barrier functions and if control limits are provided, the CBF QP may not always be
feasible. If allowing for relaxation in the CBFConfig, a slack variable will be introduced to ensure that the
problem is always feasible, with a high penalty on any infeasibility. This is generally useful for controller
robustness, but means that safety is not guaranteed.

If strict enforcement of the CBF is desired, your higest-level controller should handle the case where the QP
is infeasible.
"""

from typing import Optional, Callable
from abc import ABC, abstractmethod
import warnings

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


class CBFConfig(ABC):
    """Control Barrier Function (CBF) configuration class.

    This is an abstract class which requires implementation of the following methods:

    - `f(z)`: The uncontrolled dynamics function
    - `g(z)`: The control affine dynamics function
    - `h_1(z)` and/or `h_2(z)`: The barrier function(s), of relative degree 1 and/or 2

    For finer-grained control over the CBF, the following methods may be updated from their defaults:

    - `alpha(h)`: "Gain" of the CBF
    - `alpha_2(h_2)`: "Gain" of the relative-degree-2 CBFs, if applicable
    - `P(z, u_des)`: Quadratic term in the CBF QP objective
    - `q(z, u_des)`: Linear term in the CBF QP objective

    Args:
        n (int): State dimension
        m (int): Control dimension
        u_min (ArrayLike, optional): Minimum control input, shape (m,). Defaults to None (Unconstrained).
        u_max (ArrayLike, optional): Maximum control input, shape (m,). Defaults to None (Unconstrained).
        relax_qp (bool, optional): Whether to allow for relaxation in the CBF QP. Defaults to True.
            Note: this is required for differentiability through the QP.
        cbf_relaxation_penalty (float, optional): Penalty on the CBF slack variables in the relaxed QP.
            Defaults to 1e3. Note: only applies if relax_qp is True.
        control_relaxation_penalty (float, optional): Penalty on the control constraint slack variables in the
            relaxed QP. Defaults to 1e5. Note: only applies if relax_qp is True.
        solver_tol (float, optional): Tolerance for the QP solver. Defaults to 1e-3.
        init_args (tuple, optional): If your barrier function relies on additional arguments other than just the state,
            include an initial seed for these arguments here. This is to help test the output of the barrier function.
            Defaults to ().
    """

    def __init__(
        self,
        n: int,
        m: int,
        u_min: Optional[ArrayLike] = None,
        u_max: Optional[ArrayLike] = None,
        relax_qp: bool = True,
        cbf_relaxation_penalty: float = 1e3,
        control_relaxation_penalty: float = 1e5,
        solver_tol: float = 1e-3,
        init_args: tuple = (),
    ):
        if not (isinstance(n, int) and n > 0):
            raise ValueError(f"n must be a positive integer. Got: {n}")
        self.n = n

        if not (isinstance(m, int) and m > 0):
            raise ValueError(f"m must be a positive integer. Got: {m}")
        self.m = m

        if not isinstance(relax_qp, bool):
            raise ValueError(f"relax_qp must be a boolean. Got: {relax_qp}")
        self.relax_qp = relax_qp

        if not (
            isinstance(cbf_relaxation_penalty, (int, float))
            and cbf_relaxation_penalty >= 0
        ):
            raise ValueError(
                f"CBF relaxation penalty must be a non-negative value. Got: {cbf_relaxation_penalty}"
            )
        self.cbf_relaxation_penalty = float(cbf_relaxation_penalty)

        if not (isinstance(solver_tol, (int, float)) and solver_tol > 0):
            raise ValueError(f"solver_tol must be a positive value. Got: {solver_tol}")
        self.solver_tol = float(solver_tol)

        if self.solver_tol > 1e-2:
            print(
                f"WARNING: solver tolerance is quite high ({self.solver_tol}). "
                + " Solution will likely be poor."
            )

        if not isinstance(init_args, tuple):
            raise ValueError(f"init_args must be a tuple. Got: {init_args}")
        self.init_args = init_args

        if not (
            isinstance(control_relaxation_penalty, (int, float))
            and control_relaxation_penalty >= 0
        ):
            raise ValueError(
                f"control_relaxation_penalty must be a non-negative value. Got: {control_relaxation_penalty}"
            )
        self.control_relaxation_penalty = float(control_relaxation_penalty)

        # Control limits require a bit of extra handling. They can be both None if unconstrained,
        # but we should not have one limit as None and the other as some value
        u_min = np.asarray(u_min, dtype=float).flatten() if u_min is not None else None
        u_max = np.asarray(u_max, dtype=float).flatten() if u_max is not None else None
        if u_min is not None or u_max is not None:
            self.control_constrained = True
            if u_min is None and u_max is not None:
                u_min = -np.inf * np.ones(self.m)
            elif u_min is not None and u_max is None:
                u_max = np.inf * np.ones(self.m)
        else:
            self.control_constrained = False
        if u_min is not None:
            assert u_min.shape == (self.m,)
            u_min = tuple(u_min)
        if u_max is not None:
            assert u_max.shape == (self.m,)
            u_max = tuple(u_max)
        self.u_min = u_min
        self.u_max = u_max

        if (
            self.control_constrained
            and self.control_relaxation_penalty < self.cbf_relaxation_penalty
        ):
            print("WARNING: Control constraints have a lower penalty than the CBFs.")

        # Test if the methods are provided and verify their output dimension
        z_test = jnp.ones(self.n)
        u_test = jnp.ones(self.m)
        f_test = self.f(z_test)
        g_test = self.g(z_test)
        if f_test.shape != (self.n,):
            raise ValueError(
                f"Invalid shape for f(z). Got {f_test.shape}, expected ({self.n},)"
            )
        if g_test.shape != (self.n, self.m):
            raise ValueError(
                f"Invalid shape for g(z). Got {g_test.shape}, expected ({self.n}, {self.m})"
            )
        try:
            h1_test = self.h_1(z_test, *self.init_args)
            h2_test = self.h_2(z_test, *self.init_args)
        except TypeError as e:
            raise ValueError(
                "Cannot test the barrier function; likely missing additional arguments.\n"
                + "Please provide an initial seed for these args in the config's init_args input"
            ) from e
        if h1_test.ndim != 1 or h2_test.ndim != 1:
            raise ValueError("Barrier function(s) must be 1D arrays")
        self.num_rd1_cbf = h1_test.shape[0]
        self.num_rd2_cbf = h2_test.shape[0]
        self.num_cbf = self.num_rd1_cbf + self.num_rd2_cbf
        if self.num_cbf == 0:
            raise ValueError(
                "No barrier functions provided."
                + "\nYou can implement this via the h_1 and/or h_2 methods in your config class"
            )
        h_test = jnp.concatenate([h1_test, h2_test])
        alpha_test = self.alpha(h_test)
        alpha_2_test = self.alpha_2(h2_test)
        if alpha_test.shape != (self.num_cbf,):
            raise ValueError(
                f"Invalid shape for alpha(h(z)): {alpha_test.shape}. Expected ({self.num_cbf},)"
                + "\nCheck that the output of the alpha() function matches the number of CBFs"
            )
        if alpha_2_test.shape != (self.num_rd2_cbf,):
            raise ValueError(
                f"Invalid shape for alpha_2(h_2(z)): {alpha_2_test.shape}. Expected ({self.num_rd2_cbf},)"
                + "\nCheck that the output of the alpha_2() function matches the number of RD2 CBFs"
            )
        self._check_class_kappa(self.alpha, self.num_cbf)
        self._check_class_kappa(self.alpha_2, self.num_rd2_cbf)
        try:
            P_test = self.P(z_test, u_test, *self.init_args)
        except TypeError as e:
            raise ValueError(
                "Cannot test the P matrix; likely missing additional arguments.\n"
                + "Please provide an initial seed for these args in the config's init_args input"
            ) from e
        if P_test.shape != (self.m, self.m):
            raise ValueError(
                f"Invalid shape for P(z). Got {P_test.shape}, expected ({self.m}, {self.m})"
            )
        if not self._is_symmetric_psd(P_test):
            raise ValueError("P matrix must be symmetric positive semi-definite")

        # Handle QP relaxation penalties, if relaxation is enabled
        num_qp_constraints = (
            self.num_cbf if not self.control_constrained else self.num_cbf + 2 * self.m
        )
        if self.control_constrained:
            self.constraint_relaxation_penalties = tuple(
                np.concatenate(
                    [
                        self.cbf_relaxation_penalty * np.ones(self.num_cbf),
                        self.control_relaxation_penalty * np.ones(2 * self.m),
                    ]
                )
            )
        else:
            self.constraint_relaxation_penalties = tuple(
                self.cbf_relaxation_penalty * np.ones(self.num_cbf)
            )
        assert len(self.constraint_relaxation_penalties) == num_qp_constraints

    ## Control Affine Dynamics ##

    @abstractmethod
    def f(self, z: ArrayLike) -> Array:
        """The uncontrolled dynamics function. Possibly nonlinear, and locally Lipschitz

        i.e. the function f, such that z_dot = f(z) + g(z) u

        Args:
            z (ArrayLike): The state, shape (n,)

        Returns:
            Array: Uncontrolled state derivative component, shape (n,)
        """
        pass

    @abstractmethod
    def g(self, z: ArrayLike) -> Array:
        """The control affine dynamics function. Locally Lipschitz.

        i.e. the function g, such that z_dot = f(z) + g(z) u

        Args:
            z (ArrayLike): The state, shape (n,)

        Returns:
            Array: Control matrix, shape (n, m)
        """
        pass

    ## Barriers ##

    def h_1(self, z: ArrayLike, *h_args) -> Array:
        """Relative-degree-1 barrier function(s).

        A (zeroing) CBF is a continuously-differentiable function h, such that for any state z in the interior of
        the safe set, h(z) should be > 0, and h(z) = 0 on the boundary. When in the unsafe set, h(z) < 0.

        Relative degree can generally be thought of as the number of integrations required between the
        input and output of the system. For instance, a (relative-degree-1) CBF based on velocities,
        with acceleration inputs, will be directly modified on the next timestep.

        If your barrier function is relative-degree-2, or if you would like to enforce additional barriers
        which are relative-degree-2, use the `h_2` method.

        Args:
            z (ArrayLike): State, shape (n,)
            *h_args: Optional additional arguments for the barrier function. Note: If using additional args with your
                barrier, these must be a static shape/type, or else this will trigger a recompilation in Jax.

        Returns:
            Array: Barrier function(s), shape (num_rd1_barr,)
        """
        return jnp.array([])

    def h_2(self, z: ArrayLike, *h_args) -> Array:
        """Relative-degree-2 (high-order) barrier function(s).

        A (zeroing) CBF is a continuously-differentiable function h, such that for any state z in the interior of
        the safe set, h(z) should be > 0, and h(z) = 0 on the boundary. When in the unsafe set, h(z) < 0.

        Relative degree can generally be thought of as the number of integrations required between the
        input and output of the system. For instance, a (relative-degree-2) CBF based on position,
        with acceleration inputs, will be modified in two timesteps: the acceleration changes the velocity,
        which then changes the position.

        If your barrier function is relative-degree-1, or if you would like to enforce additional barriers
        which are relative-degree-1, use the `h_1` method.

        Args:
            z (ArrayLike): State, shape (n,)
            *h_args: Optional additional arguments for the barrier function. Note: If using additional args with your
                barrier, these must be a static shape/type, or else this will trigger a recompilation in Jax.

        Returns:
            Array: Barrier function(s), shape (num_rd2_barr,)
        """
        return jnp.array([])

    ## Additional tuning functions ##

    def alpha(self, h: ArrayLike) -> Array:
        """A class Kappa function, dictating the "gain" of the barrier function(s)

        For reference, a class Kappa function is a monotonically increasing function which passes through the origin.
        A simple example is alpha(h) = h

        The default implementation can be overridden for more fine-grained control over the CBF

        Args:
            h (ArrayLike): Evaluation of the barrier function(s) at the current state, shape (num_cbf,)

        Returns:
            Array: alpha(h(z)), shape (num_cbf,)
        """
        return h

    def alpha_2(self, h_2: ArrayLike) -> Array:
        """A second class Kappa function which dictactes the "gain" associated with the relative-degree-2
        barrier functions

        For reference, a class Kappa function is a monotonically increasing function which passes through the origin.
        A simple example is alpha_2(h_2) = h_2

        The default implementation can be overridden for more fine-grained control over the CBF

        Args:
            h_2 (ArrayLike): Evaluation of the RD2 barrier function(s) at the current state, shape (num_rd2_cbf,)

        Returns:
            Array: alpha_2(h_2(z)), shape (num_rd2_cbf,).
        """
        return h_2

    # Objective function tuning

    def P(self, z: Array, u_des: Array, *h_args) -> Array:
        """Quadratic term in the CBF QP objective (minimize 0.5 * x^T P x + q^T x)

        This defaults to 2 * I, which is the value of P when minimizing the standard CBF objective,
        ||u - u_des||_{2}^{2}

        To change the objective, override this method. **Note that P must be PSD**

        Args:
            z (Array): State, shape (n,)
            u_des (Array): Desired control input, shape (m,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            Array: P matrix, shape (m, m)
        """
        return 2 * jnp.eye(self.m)

    def q(self, z: Array, u_des: Array, *h_args) -> Array:
        """Linear term in the CBF QP objective (minimize 0.5 * x^T P x + q^T x)

        This defaults to -2 * u_des, which is the value of q when minimizing the standard CBF objective,
        ||u - u_des||_{2}^{2}

        To change the objective, override this method.

        Args:
            z (Array): State, shape (n,)
            u_des (Array): Desired control input, shape (m,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            Array: q vector, shape (m,)
        """
        return -2 * u_des

    ## Helper functions ##

    def _check_class_kappa(
        self, func: Callable[[ArrayLike], ArrayLike], dim: int
    ) -> None:
        """Checks that the provided function is in class Kappa

        Args:
            func (Callable): Function to check
            dim (int): Expected dimension of the output
        """
        assert isinstance(func, Callable)
        try:
            # Check that func(0) == 0
            assert jnp.allclose(func(jnp.zeros(dim)), 0.0)
            # Check that func is monotonically increasing
            n_test = 100
            test_points = jnp.repeat(
                jnp.linspace(-1e6, 1e6, n_test).reshape(n_test, 1), dim, axis=1
            )
            a = jax.vmap(func, in_axes=0)(test_points)
            assert jnp.all(a[:-1, :] < a[1:, :])
        except AssertionError as e:
            raise ValueError(
                f"{func.__name__} does not appear to be a class Kappa function"
            ) from e

    def _is_symmetric_psd(self, mat: Array) -> bool:
        """Check that a matrix is symmetric positive semi-definite

        Args:
            mat (Array): Matrix to check

        Returns:
            bool: True if the matrix is symmetric PSD, False otherwise
        """
        mat = np.asarray(mat)
        # Must be square
        if mat.shape[0] != mat.shape[1]:
            return False
        # Must be symmetric or hermitian
        if not np.allclose(mat, mat.conj().T, atol=1e-14):
            return False
        # Check PSD with cholesky. Cholesky can only tell if a matrix is PD, not PSD,
        # but adding a small regularization term will allow this test to work
        try:
            np.linalg.cholesky(mat + np.eye(mat.shape[0]) * 1e-14)
        except np.linalg.LinAlgError:
            return False
        return True
