"""
# Base Environment

This is a convenient structure for building demo environments to test CBFs. However, it is not necessary to use this.

For instance, going back to the CBF usage pseudocode,
```
while True:
    z = get_state()
    z_des = get_desired_state()
    u_nom = nominal_controller(z, z_des)
    u = cbf.safety_filter(z, u_nom)
    apply_control(u)
    step()
```
We use this base environment to set up the `get_state`, `get_desired_state`, `apply_control`, and `step` methods in
any derived environments.
"""

from abc import ABC, abstractmethod

from jax.typing import ArrayLike


class BaseEnv(ABC):
    """Simulation environment Abstract Base Class for testing CBFs

    Any environment inheriting from this class should implement the following methods:

    - `step`: Run a single simulation step
    - `get_state`: Get the current state of the robot
    - `get_desired_state`: Get the desired state of the robot
    - `apply_control`: Apply a control input to the robot
    """

    @abstractmethod
    def step(self) -> None:
        """Runs a single simulation step for the environment

        This should update any dynamics and visuals
        """
        pass

    @abstractmethod
    def get_state(self) -> ArrayLike:
        """Returns the current state of the environment

        Returns:
            ArrayLike: State, shape (n,)
        """
        pass

    @abstractmethod
    def get_desired_state(self) -> ArrayLike:
        """Returns the desired state of the environment

        Returns:
            ArrayLike: Desired state, shape (n,)
        """
        pass

    @abstractmethod
    def apply_control(self, u: ArrayLike) -> None:
        """Applies the control input to the environment

        Args:
            u (ArrayLike): Control, shape (m,)
        """
        pass
