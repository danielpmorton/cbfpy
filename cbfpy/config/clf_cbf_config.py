"""
# CLF-CBF Configuration class

## Defining the problem:

As with the CBF, we require implementation of the dynamics functions `f` and `g`, as well as the barrier function(s)
`h`. Now, with the CLF-CBF, we require the definition of the Control Lyapunov Function (CLF) `V`. This CLF must be a
positive definite function of the state.

Depending on the relative degree of your barrier function(s), you should implement the `h_1` method
(for a relative-degree-1 barrier), and/or the `h_2` method (for a relative-degree-2 barrier).

Likewise, for the CLF, you should implement the `V_1` method (for a relative-degree-1 CLF), and/or the `V_2` method
(for a relative-degree-2 CLF).

## Tuning the CLF-CBF:

As with the CBF, the CLF-CBF config allows for adjustment of the class-Kappa CBF "gain" functions `alpha` and `alpha_2`.
Additionally, the CLF-CBF config allows for adjustment of the class-Kappa CLF "gain" functions `gamma` and `gamma_2`
(for relative-degree-2 CLFs).

The CLF-CBF config also allows for adjustment of the quadratic control term `H` and the linear control term `F` in the
CLF objective. These can be used to adjust the weightings between inputs, for instance.

## Relaxation:

If the CBF constraints are not necessarily globally feasible, you can enable further relaxation in the CLFCBFConfig.
However, since the CLF constraint was already relaxed with respect to the CBF constraint, this means that tuning the
relaxation parameters is critical. In general, the penalty on the CBF relaxation should be much higher than the penalty
on the CLF relaxation.

If strict enforcement of the CLF-CBF is desired, your higest-level controller should handle the case where the QP
is infeasible.
"""

from typing import Optional

import numpy as np
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from cbfpy.config.cbf_config import CBFConfig


class CLFCBFConfig(CBFConfig):
    """Control Lyapunov Function / Control Barrier Function (CLF-CBF) configuration class.

    This is an abstract class which requires implementation of the following methods:

    - `f(z)`: The uncontrolled dynamics function
    - `g(z)`: The control affine dynamics function
    - `h_1(z)` and/or `h_2(z)`: The barrier function(s), of relative degree 1 and/or 2
    - `V_1(z)` and/or `V_2(z)`: The Lyapunov function(s), of relative degree 1 and/or 2

    For finer-grained control over the CLF-CBF, the following methods may be updated from their defaults:

    - `alpha(h)`: "Gain" of the CBF
    - `alpha_2(h_2)`: "Gain" of the relative-degree-2 CBFs, if applicable
    - `gamma(v)`: "Gain" of the CLF
    - `gamma_2(v)`: "Gain" of the relative-degree-2 CLFs, if applicable
    - `H(z)`: Quadratic control term in the CLF objective
    - `F(z)`: Linear control term in the CLF objective

    Args:
        n (int): State dimension
        m (int): Control dimension
        u_min (ArrayLike, optional): Minimum control input, shape (m,). Defaults to None (Unconstrained).
        u_max (ArrayLike, optional): Maximum control input, shape (m,). Defaults to None (Unconstrained).
        relax_qp (bool, optional): Whether to allow for relaxation in the CLF-CBF QP. Defaults to True.
            Note: this is required for differentiability through the QP.
        cbf_relaxation_penalty (float, optional): Penalty on the slack variable in the relaxed QP. Defaults to 1e4.
            Note: only applies if relax_qp is True.
        clf_relaxation_penalty (float): Penalty on the CLF slack variable when enforcing the CBF. Defaults to 1e2
        control_relaxation_penalty (float, optional): Penalty on the control constraint slack variables in the
            relaxed QP. Defaults to 1e5. Note: only applies if relax_qp is True.
        solver_tol (float, optional): Tolerance for the QP solver. Defaults to 1e-3.
        init_args (tuple, optional): If your barriers or dynamics rely on additional (non-differentiable, static shape)
            args other than just the state, include an initial seed for these args here. Defaults to None.
        init_kwargs (dict, optional): If your barriers or dynamics rely on additional (non-differentiable, static shape)
            kwargs other than just the state, include an initial seed for these kwargs here. Defaults to None.
    """

    def __init__(
        self,
        n: int,
        m: int,
        u_min: Optional[ArrayLike] = None,
        u_max: Optional[ArrayLike] = None,
        relax_qp: bool = True,
        cbf_relaxation_penalty: float = 1e4,
        clf_relaxation_penalty: float = 1e2,
        control_relaxation_penalty: float = 1e5,
        solver_tol: float = 1e-3,
        init_args: Optional[tuple] = None,
        init_kwargs: Optional[dict] = None
    ):
        super().__init__(
            n,
            m,
            u_min,
            u_max,
            relax_qp,
            cbf_relaxation_penalty,
            control_relaxation_penalty,
            solver_tol,
            init_args,
            init_kwargs,
        )

        if not (
            isinstance(clf_relaxation_penalty, (int, float))
            and clf_relaxation_penalty > 0
        ):
            raise ValueError(
                f"Invalid clf_relaxation_penalty: {clf_relaxation_penalty}. Must be a positive value."
            )
        self.clf_relaxation_penalty = float(clf_relaxation_penalty)

        # If relaxing the QP, need to have CLF penalty < CBF penalty < Control constraint penalty
        if self.cbf_relaxation_penalty < self.clf_relaxation_penalty:
            print("WARNING: CBF constraints have a lower penalty than the CLFs")
        if (
            self.control_constrained
            and self.control_relaxation_penalty < self.clf_relaxation_penalty
        ):
            print("WARNING: Control constraints have a lower penalty than the CLFs")

        # Check on CLF dimension
        z_test = jnp.ones(self.n)
        v1_test = self.V_1(z_test, z_test, *self.init_args, **self.init_kwargs)
        v2_test = self.V_2(z_test, z_test, *self.init_args, **self.init_kwargs)
        if v1_test.ndim != 1 or v2_test.ndim != 1:
            raise ValueError("CLF(s) must output 1D arrays")
        self.num_rd1_clf = v1_test.shape[0]
        self.num_rd2_clf = v2_test.shape[0]
        self.num_clf = self.num_rd1_clf + self.num_rd2_clf
        if self.num_clf == 0:
            raise ValueError(
                "No Lyanpunov functions provided."
                + "\nYou can implement this via the V_1 and/or V_2 methods in your config class"
            )
        v_test = jnp.concatenate([v1_test, v2_test])
        gamma_test = self.gamma(v_test, *self.init_args, **self.init_kwargs)
        gamma_2_test = self.gamma_2(v2_test, *self.init_args, **self.init_kwargs)
        if gamma_test.shape != (self.num_clf,):
            raise ValueError(
                f"Invalid shape for gamma(V(z)): {gamma_test.shape}. Expected ({self.num_clf},)"
                + "\nCheck that the output of the gamma() function matches the number of CLFs"
            )
        if gamma_2_test.shape != (self.num_rd2_clf,):
            raise ValueError(
                f"Invalid shape for gamma_2(V_2(z)): {gamma_2_test.shape}. Expected ({self.num_rd2_clf},)"
                + "\nCheck that the output of the gamma_2() function matches the number of RD2 CLFs"
            )
        self._check_class_kappa(self.gamma, self.num_clf, *self.init_args, **self.init_kwargs)
        self._check_class_kappa(self.gamma_2, self.num_rd2_clf, *self.init_args, **self.init_kwargs)
        H_test = self.H(z_test, *self.init_args, **self.init_kwargs)
        if H_test.shape != (self.m, self.m):
            raise ValueError(
                f"Invalid shape for H(z): {H_test.shape}. Expected ({self.m}, {self.m})"
            )
        if not self._is_symmetric_psd(H_test):
            raise ValueError("H(z) must be symmetric positive semi-definite")

        # Handle QP relaxation penalties, if relaxation is enabled
        num_qp_constraints = (
            self.num_cbf + self.num_clf
            if not self.control_constrained
            else self.num_cbf + self.num_clf + 2 * self.m
        )
        if self.control_constrained:
            self.constraint_relaxation_penalties = tuple(
                np.concatenate(
                    [
                        self.clf_relaxation_penalty * np.ones(self.num_clf),
                        self.cbf_relaxation_penalty * np.ones(self.num_cbf),
                        self.control_relaxation_penalty * np.ones(2 * self.m),
                    ]
                )
            )
        else:
            self.constraint_relaxation_penalties = tuple(
                np.concatenate(
                    [
                        self.clf_relaxation_penalty * np.ones(self.num_clf),
                        self.cbf_relaxation_penalty * np.ones(self.num_cbf),
                    ]
                )
            )
        assert len(self.constraint_relaxation_penalties) == num_qp_constraints

    def V_1(self, z: ArrayLike, z_des: ArrayLike, *args, **kwargs) -> Array:
        """Relative-Degree-1 Control Lyapunov Function (CLF)

        A CLF is a positive-definite function which evaluates to zero at the equilibrium point, and is
        such that there exists a control input u which makes the time-derivative of the CLF negative.

        Relative degree can generally be thought of as the number of integrations required between the
        input and output of the system. For instance, a (relative-degree-1) CLF based on velocities,
        with acceleration inputs, will be directly modified on the next timestep.

        At least one of `V_1` or `V_2` must be implemented. Multiple CLFs is possible, but generally, these cannot all
        be strictly enforced.

        Args:
            z (ArrayLike): State, shape (n,)
            z_des (ArrayLike): Desired state, shape (n,)

        Returns:
            Array: V(z): The RD1 CLF evaluation, shape (num_rd1_clf,)
        """
        return jnp.array([])

    # TODO: Check if the math behind this is actually valid
    def V_2(self, z: ArrayLike, z_des: ArrayLike, *args, **kwargs) -> Array:
        """Relative-Degree-2 (high-order) Control Lyapunov Function (CLF)

        A CLF is a positive-definite function which evaluates to zero at the equilibrium point, and is
        such that there exists a control input u which makes the time-derivative of the CLF negative.

        Relative degree can generally be thought of as the number of integrations required between the
        input and output of the system. For instance, a (relative-degree-2) CLF based on position,
        with acceleration inputs, will be modified in two timesteps: the acceleration changes the velocity,
        which then changes the position.

        At least one of `V_1` or `V_2` must be implemented. Multiple CLFs is possible, but generally, these cannot all
        be strictly enforced.

        Args:
            z (ArrayLike): State, shape (n,)
            z_des (ArrayLike): Desired state, shape (n,)

        Returns:
            Array: V(z): The RD2 CLF evaluation, shape (num_rd2_clf,)
        """
        return jnp.array([])

    def gamma(self, v: ArrayLike, *args, **kwargs) -> Array:
        """A class Kappa function, dictating the "gain" of the CLF

        For reference, a class Kappa function is a monotonically increasing function which passes through the origin.

        The default implementation can be overridden for more fine-grained control over the CLF

        Args:
            v (ArrayLike): Evaluation of the CLF(s) at the current state, shape (num_clf,).

        Returns:
            Array: gamma(V(z)), shape (num_clf,).
        """
        return v

    def gamma_2(self, v_2: ArrayLike, *args, **kwargs) -> Array:
        """A second class Kappa function, dictating the "gain" associated with the derivative of the CLF

        For reference, a class Kappa function is a monotonically increasing function which passes through the origin.

        The default implementation can be overridden for more fine-grained control over the CLF

        Args:
            v_2 (ArrayLike): Evaluation of the RD2 CLF(s) at the current state, shape (num_rd2_clf,)

        Returns:
            Array: gamma_2(V_2(z)), shape (num_rd2_clf,)
        """
        return v_2

    def H(self, z: ArrayLike, *args, **kwargs) -> Array:
        """Matrix defining the quadratic control term in the CLF objective (minimize 0.5 * u^T H u + F^T u)

        **Must be PSD!**

        The default implementation is just the (m x m) identity matrix, but this can be overridden
        for more fine-grained control over the objective

        Args:
            z (ArrayLike): State, shape (n,)

        Returns:
            Array: H, shape (m, m)
        """
        return jnp.eye(self.m)

    def F(self, z: ArrayLike, *args, **kwargs) -> Array:
        """Vector defining the linear term in the CLF objective (minimize 0.5 * u^T H u + F^T u)

        The default implementation is a zero vector, but this can be overridden
        for more fine-grained control over the objective

        Args:
            z (ArrayLike): State, shape (n,)

        Returns:
            Array: F, shape (m,)
        """
        return jnp.zeros(self.m)
