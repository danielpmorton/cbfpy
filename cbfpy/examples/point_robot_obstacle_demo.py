"""
# Point Robot Obstacle Avoidance Demo

Example of using a CBF + PD controller to control a point robot to reach the origin,
while avoiding a moving obstacle and staying in a safe set.

This demo is interactive: click and drag the obstacle to move it around.

Here, we have double integrator dynamics: z = [position, velocity], u = [acceleration]
and we also use the state of the obstacle as an input to the CBF: z_obs = [position, velocity]

This example includes both relative-degree-1 and relative-degree-2 CBFs. Staying inside the safe-set box is
RD2, since we have a positional barrier with acceleration inputs. Avoiding the obstacle is relative-degree-1,
because this is based on the relative velocity between the two objects.
"""

import time
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

from cbfpy import CBF, CBFConfig
from cbfpy.envs.point_robot_envs import PointRobotObstacleEnv


class PointRobotObstacleConfig(CBFConfig):
    """Configuration for the 3D 'point-robot-avoiding-an-obstacle' example."""

    def __init__(self):
        self.mass = 1.0
        self.robot_radius = 0.25
        self.obstacle_radius = 0.25
        init_z_obs = jnp.array([3.0, 1.0, 1.0, -0.1, -0.1, -0.1])
        super().__init__(
            n=6,  # State = [position, velocity]
            m=3,  # Control = [force]
            init_args=(init_z_obs,),
        )

    def f(self, z, *args, **kwargs):
        A = jnp.block(
            [[jnp.zeros((3, 3)), jnp.eye(3)], [jnp.zeros((3, 3)), jnp.zeros((3, 3))]]
        )
        return A @ z

    def g(self, z, *args, **kwargs):
        B = jnp.block([[jnp.zeros((3, 3))], [jnp.eye(3) / self.mass]])
        return B

    def h_1(self, z, z_obs, **kwargs):
        # Distance between >= obstacle radius + robot radius + deceleration distance
        pos_robot = z[:3]
        vel_robot = z[3:]
        pos_obs = z_obs[:3]
        vel_obs = z_obs[3:]
        dist_between_centers = jnp.linalg.norm(pos_obs - pos_robot)
        dir_obs_to_robot = (pos_robot - pos_obs) / dist_between_centers
        collision_velocity_component = (vel_obs - vel_robot).T @ dir_obs_to_robot
        lookahead_time = 2.0
        padding = 0.1
        return jnp.array(
            [
                dist_between_centers
                - collision_velocity_component * lookahead_time
                - self.obstacle_radius
                - self.robot_radius
                - padding
            ]
        )

    def h_2(self, z, z_obs, **kwargs):
        # Stay inside the safe set (a box)
        pos_max = jnp.array([1.0, 1.0, 1.0])
        pos_min = jnp.array([-1.0, -1.0, -1.0])
        return jnp.concatenate([pos_max - z[:3], z[:3] - pos_min])

    def alpha(self, h, *args, **kwargs):
        return 3 * h


@jax.jit
def nominal_controller(z: ArrayLike, z_des: ArrayLike) -> Array:
    """A simple PD controller for the point robot.

    This is unsafe without the CBF, as there is no guarantee that the robot wil not collide with the obstacle

    Args:
        z (ArrayLike): The current state of the robot [x, y, z, vx, vy, vz]
        z_des (ArrayLike): The desired state of the robot [x_des, y_des, z_des, vx_des, vy_des, vz_des]
    """
    Kp = 1.0
    Kd = 1.0
    u = -Kp * (z[:3] - z_des[:3]) - Kd * (z[3:] - z_des[3:])
    return u


def main():
    config = PointRobotObstacleConfig()
    cbf = CBF.from_config(config)
    env = PointRobotObstacleEnv()

    @jax.jit
    def safe_controller(z, z_des, z_obs):
        u = nominal_controller(z, z_des)
        return cbf.safety_filter(z, u, z_obs)

    while True:
        z, z_obs = env.get_state()
        z_des = env.get_desired_state()
        u = safe_controller(z, z_des, z_obs)
        env.apply_control(u)
        env.step()
        time.sleep(1 / 300)


if __name__ == "__main__":
    main()
