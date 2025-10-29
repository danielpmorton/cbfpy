"""
# Drone Environment

This is a wrapper around the gym-pybullet-drones environment, using velocity control.
"""

import time
import warnings

import numpy as np
import pybullet
from pybullet_utils.bullet_client import BulletClient
from jax.typing import ArrayLike
from jax import Array

# Note: the original gym_pybullet_drones repo has a lot of dependencies that are not necessary for this demo.
# Use the fork at https://github.com/danielpmorton/gym-pybullet-drones instead
try:
    from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
except ImportError as e:
    raise ImportError(
        "Please install the forked version of gym-pybullet-drones:"
        + "\n'pip install 'gym_pybullet_drones @ git+https://github.com/danielpmorton/gym-pybullet-drones.git''"
    ) from e

from cbfpy.utils.visualization import visualize_3D_box
from cbfpy.envs.base_env import BaseEnv
from cbfpy.utils.general_utils import find_assets_dir, stdout_redirected


class DroneEnv(BaseEnv):
    """Drone Environment class

    This provides an environment where the drone is contained to lie within a safe box,
    and must avoid a movable obstacle.

    Args:
        xyz_min (ArrayLike): Minimum bounds of the safe region, shape (3,)
        xyz_max (ArrayLike): Maximum bounds of the safe region, shape (3,)
    """

    # Constants
    RADIUS = 0.1  # TODO tune

    def __init__(
        self,
        xyz_min: ArrayLike = (-0.5, -0.5, 0.5),
        xyz_max: ArrayLike = (0.5, 0.5, 1.5),
    ):
        # Suppress pybullet output + a Gym warning about float32 precision
        with stdout_redirected(restore=False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._env = VelocityAviary(
                    # drone_model="cf2x",
                    num_drones=1,
                    initial_xyzs=np.array([[0, 0, 1]]),
                    initial_rpys=np.array([[0, 0, 0]]),
                    # physics="pyb",
                    neighbourhood_radius=np.inf,
                    pyb_freq=240,
                    ctrl_freq=48,
                    gui=True,
                    record=False,
                    obstacles=False,
                    user_debug_gui=False,
                )
            # Hack: Create same client interface as other envs
            self.client: pybullet = object.__new__(BulletClient)
            self.client._client = self._env.CLIENT
        self.xyz_min = np.array(xyz_min)
        self.xyz_max = np.array(xyz_max)
        assert len(self.xyz_min) == len(self.xyz_max) == 3
        self.client.setAdditionalSearchPath(find_assets_dir())
        self.obstacle = self.client.loadURDF("point_robot.urdf", basePosition=(1, 1, 1))
        self.client.configureDebugVisualizer(self.client.COV_ENABLE_GUI, 0)
        self.client.resetDebugVisualizerCamera(1.80, 37.60, -25.00, (0.05, 0.03, 0.75))
        self.robot = self._env.DRONE_IDS[0]
        self.client.changeVisualShape(self.obstacle, -1, rgbaColor=[1, 0, 0, 1])
        self.client.changeDynamics(self.obstacle, -1, angularDamping=10)
        self.box = visualize_3D_box(
            [self.xyz_min - self.RADIUS, self.xyz_max + self.RADIUS],
            rgba=(0, 1, 0, 0.5),
        )
        # For color determination
        self.is_in_box = True
        self.tol = 1e-3

        self.action = np.array([[0.0, 0.0, 0.0, 0.0]])
        self.obs, self.reward, self.terminated, self.truncated, self.info = (
            self._env.step(self.action)
        )

    def _update_color(self, robot_pos: ArrayLike) -> None:
        """Update the color of the box depending on if the robot is inside or not (Green inside, red outside)"""
        robot_pos = np.array(robot_pos)
        if np.any(robot_pos < self.xyz_min - self.tol) or np.any(
            robot_pos > self.xyz_max + self.tol
        ):
            if self.is_in_box:
                self.client.changeVisualShape(self.box, -1, rgbaColor=[1, 0, 0, 0.5])
            self.is_in_box = False
        else:
            if not self.is_in_box:
                self.client.changeVisualShape(self.box, -1, rgbaColor=[0, 1, 0, 0.5])
            self.is_in_box = True

    def get_state(self) -> Array:
        robot_pos = self.client.getBasePositionAndOrientation(self.robot)[0]
        robot_vel = self.client.getBaseVelocity(self.robot)[0]
        self._update_color(robot_pos)
        obstacle_pos = self.client.getBasePositionAndOrientation(self.obstacle)[0]
        obstacle_vel = self.client.getBaseVelocity(self.obstacle)[0]
        return np.array([*robot_pos, *robot_vel]), np.array(
            [*obstacle_pos, *obstacle_vel]
        )

    def get_desired_state(self) -> Array:
        return np.array([0, 0, 1, 0, 0, 0])

    def apply_control(self, u: Array) -> None:
        # Note: the gym-pybullet-drones API has a weird format for the "velocity action" to take
        self.action = np.array([[*u, np.linalg.norm(u)]])

    def step(self):
        # Step the gym-pybullet-drones environment using the last stored action
        self.obs, self.reward, self.terminated, self.truncated, self.info = (
            self._env.step(self.action)
        )


def _test_env():
    """Test the environment behavior under an **unsafe** nominal controller"""

    env = DroneEnv()

    def nominal_controller(z, z_des):
        Kp = 1.0
        Kv = 1.0
        return Kp * (z_des[:3] - z[:3]) + Kv * (z_des[3:] - z[3:])

    print("\nDemoing the drone environment with an unsafe controller")
    while True:
        z, z_obs = env.get_state()
        z_des = env.get_desired_state()
        u = nominal_controller(z, z_des)
        env.apply_control(u)
        env.step()
        time.sleep(1 / 300)


if __name__ == "__main__":
    _test_env()
