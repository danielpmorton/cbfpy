"""
# Point Robot Safe-Set Containment Demo

Demo of a point robot in 3D constrained to lie within a box via a CBF safety filter on a PD controller

We define the state z as [x, y, z, vx, vy, vz] and the control u as [Fx, Fy, Fz]

The dynamics are that of a simple double integrator:
z_dot = [vx, vy, vz, 0, 0, 0] + [0, 0, 0, Fx/m, Fy/m, Fz/m]

In matrix form, this can be expressed as z_dot = A z + B u with A and B as implemented in the config.

The safety constraints are set as an upper and lower bound on the position of the robot.

This is a relative-degree-2 system, so we use the RD2 version of the CBF constraints.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORMS"] = "cpu"


import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

from cbfpy import CBF, CBFConfig
from cbfpy.envs.point_robot_envs import PointRobotEnv


class PointRobotConfig(CBFConfig):
    """Configuration for the 3D 'point-robot-in-a-box' example."""

    def __init__(self):
        self.mass = 1.0
        self.pos_min = jnp.array([-1.0, -1.0, -1.0])
        self.pos_max = jnp.array([1.0, 1.0, 1.0])
        super().__init__(n=6, m=3)

    def f(self, z):
        A = jnp.block(
            [[jnp.zeros((3, 3)), jnp.eye(3)], [jnp.zeros((3, 3)), jnp.zeros((3, 3))]]
        )
        return A @ z

    def g(self, z):
        B = jnp.block([[jnp.zeros((3, 3))], [jnp.eye(3) / self.mass]])
        return B

    def h_2(self, z):
        return jnp.concatenate([self.pos_max - z[:3], z[:3] - self.pos_min])


@jax.jit
def nominal_controller(z: ArrayLike, z_des: ArrayLike) -> Array:
    """A simple PD controller for the point robot.

    This is unsafe without the CBF, as there is no guarantee that the robot will stay within the safe region

    Args:
        z (ArrayLike): The current state of the robot [x, y, z, vx, vy, vz]
        z_des (ArrayLike): The desired state of the robot [x_des, y_des, z_des, vx_des, vy_des, vz_des]
    """
    Kp = 1.0
    Kd = 1.0
    radius = 0.25
    # We assume we are in our Pybullet simulation environment where we have two point robots loaded
    # One is the robot we are controlling, and the other is the target robot, which is interactive via the GUI
    # This will attempt to "touch" the target robot
    pos_diff = z_des[:3] - z[:3]
    des_pos = z_des[:3] - 2 * radius * pos_diff / jnp.linalg.norm(pos_diff)
    u = -Kp * (z[:3] - des_pos) - Kd * (z[3:] - z_des[3:])
    return u


def main():
    config = PointRobotConfig()
    cbf = CBF.from_config(config)
    env = PointRobotEnv(config.pos_min, config.pos_max)

    @jax.jit
    def safe_controller(z, z_des):
        u = nominal_controller(z, z_des)
        return cbf.safety_filter(z, u)

    while True:
        z = env.get_state()
        z_des = env.get_desired_state()
        u = safe_controller(z, z_des)
        env.apply_control(u)
        env.step()


if __name__ == "__main__":
    main()
