"""
# Adaptive Cruise Control CLF-CBF Demo

This will enforce that the follower vehicle maintains a safe distance from the leader vehicle,
while also tracking a desired velocity.

We define the state z as [v_follower, v_leader, follow_distance] and the control u as the follower's wheel force

The dynamics incorporate a simple drag force model using empirically-derived coefficients

Note: There are a few parameters to tune in this CLF-CBF, such as the weightings between the inputs
the slack variable in the CLF objective. This is tricky to tune in general and values have been left
at what has been seen in other references.

Reference:

- "Control Barrier Function Based Quadratic Programs for Safety Critical Systems" - TAC 2017
- "Control Barrier Function based Quadratic Programs with Application to Adaptive Cruise Control" - CDC 2014

Some parameters are based on Jason Choi's https://github.com/HybridRobotics/CBF-CLF-Helper
"""

from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

from cbfpy.envs.car_env import VehicleEnv
from cbfpy.config.clf_cbf_config import CLFCBFConfig
from cbfpy.cbfs.clf_cbf import CLFCBF


class ACCConfig(CLFCBFConfig):
    """Configuration for the Adaptive Cruise Control CLF-CBF demo"""

    def __init__(self):
        self.gravity = 9.81
        self.mass = 1650.0
        self.drag_coeffs = (0.1, 5.0, 0.25)  # Drag coeffs
        self.v_des = 24.0  # Desired velocity
        self.T = 1.8  # Lookahead time
        self.cd = 0.3  # Coefficient of maximum deceleration
        self.ca = 0.3  # Coefficient of maximum acceleration
        u_min = -self.cd * self.mass * self.gravity  # Min. control input (max braking)
        u_max = self.ca * self.mass * self.gravity  # Max. control input (max throttle)
        super().__init__(
            n=3,
            m=1,
            u_min=u_min,
            u_max=u_max,
            # Note: Relaxing the CLF-CBF QP is tricky because there is an additional relaxation
            # parameter already, balancing the CLF and CBF constraints.
            relax_cbf=False,
            # If indeed relaxing, ensure that the QP relaxation >> the CLF relaxation
            clf_relaxation_penalty=10.0,
            cbf_relaxation_penalty=1e6,
        )

    def drag_force(self, v: float) -> float:
        """Compute the drag force on the follower car using a simple polynomial model

        Args:
            v (float): Velocity of the follower vehicle, in m/s

        Returns:
            float: Drag force, in Newtons
        """
        return (
            self.drag_coeffs[0] + self.drag_coeffs[1] * v + self.drag_coeffs[2] * v**2
        )

    def f(self, z: ArrayLike) -> Array:
        v_f, v_l, D = z
        # Note: We assume that the leader vehicle is at constant velocity here
        return jnp.array([-self.drag_force(v_f) / self.mass, 0.0, v_l - v_f])

    def g(self, z: ArrayLike) -> Array:
        return jnp.array([(1 / self.mass), 0.0, 0.0]).reshape(-1, 1)

    def h_1(self, z: ArrayLike) -> Array:
        v_f, v_l, D = z
        return jnp.array(
            [D - self.T * v_f - 0.5 * (v_l - v_f) ** 2 / (self.cd * self.gravity)]
        )

    def V_1(self, z: ArrayLike, z_des: ArrayLike) -> float:
        # CLF: Squared error between the follower velocity and the desired velocity
        return jnp.array([(z[0] - self.v_des) ** 2])

    def H(self, z: ArrayLike) -> Array:
        return jnp.eye(self.m) * (2 / self.mass**2)

    def F(self, z: ArrayLike) -> Array:
        return jnp.array([-2 * self.drag_force(z[0]) / self.mass**2])


def main():
    config = ACCConfig()
    clf_cbf = CLFCBF.from_config(config)
    # Ensure that parameters match up between the environment and the CLF-CBF
    env = VehicleEnv(
        "CLF-CBF Controller",
        mass=config.mass,
        drag_coeffs=config.drag_coeffs,
        v_des=config.v_des,
        u_min=config.u_min,
        u_max=config.u_max,
    )
    while env.running:
        z = env.get_state()
        z_des = env.get_desired_state()
        u = clf_cbf.controller(z, z_des)
        env.apply_control(u)
        env.step()


if __name__ == "__main__":
    main()
