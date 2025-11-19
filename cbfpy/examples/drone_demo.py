"""
# Drone Obstacle Avoidance Demo

We use velocity control, which can be easily applied to other drones
(for instance, PX4-controlled quadrotors can take in velocity commands)

The CBF uses a reduced model of the drone dynamics, while the simulation environment
does reflect a (somewhat-accurate) model of the true drone dynamics

Here, our state is the position and velocity of the drone: z = [position, velocity]
and the control input is the velocity of the drone: u = [velocity]

We also incorporate the state of the obstacle as an additional input to the CBF: z_obs = [position, velocity]

Note: since there is gravity in this simulation, the obstacle will fall to the ground initially. This is fine: just
use the mouse to drag it around near the drone to see the CBF response

See https://danielpmorton.github.io/drone_fencing/ for a demo of this on real hardware, and see the "point robot
obstacle avoidance" demo in CBFpy for a simplified version of this demo
"""

import time
import jax
import jax.numpy as jnp

from cbfpy import CBF, CBFConfig
from cbfpy.envs.drone_env import DroneEnv


class DroneConfig(CBFConfig):
    """Config for the drone obstacle avoidance / safe set containment demo"""

    def __init__(self):
        self.mass = 1.0
        self.pos_min = jnp.array([-2.0, -2.0, 0.7])
        self.pos_max = jnp.array([2.0, 2.0, 2.0])
        init_z_obs = jnp.array([3.0, 1.0, 1.0, -0.1, -0.1, -0.1])
        super().__init__(
            n=6,  # State = [position, velocity]
            m=3,  # Control = [velocity]
            relax_qp=True,
            init_args=(init_z_obs,),
            cbf_relaxation_penalty=1e6,
        )

    def f(self, z, *args, **kwargs):
        # Assume we are directly controlling the robot's velocity
        return jnp.zeros(self.n)

    def g(self, z, *args, **kwargs):
        # Assume we are directly controlling the robot's velocity
        return jnp.block([[jnp.eye(3)], [jnp.zeros((3, 3))]])

    def h_1(self, z, z_obs, **kwargs):
        obstacle_radius = 0.25
        robot_radius = 0.25
        padding = 0.1
        pos_robot = z[:3]
        vel_robot = z[3:]
        pos_obs = z_obs[:3]
        vel_obs = z_obs[3:]
        dist_between_centers = jnp.linalg.norm(pos_obs - pos_robot)
        dir_obs_to_robot = (pos_robot - pos_obs) / dist_between_centers
        collision_velocity_component = (vel_obs - vel_robot).T @ dir_obs_to_robot
        lookahead_time = 2.0
        padding = 0.1
        h_obstacle_avoidance = jnp.array(
            [
                dist_between_centers
                - collision_velocity_component * lookahead_time
                - obstacle_radius
                - robot_radius
                - padding
            ]
        )
        h_box_containment = jnp.concatenate(
            [self.pos_max - z[:3], z[:3] - self.pos_min]
        )
        return jnp.concatenate([h_obstacle_avoidance, h_box_containment])

    def alpha(self, h, *args, **kwargs):
        return jnp.array([3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * h


def nominal_controller(z, z_des):
    Kp = 1.0
    Kv = 1.0
    return Kp * (z_des[:3] - z[:3]) + Kv * (z_des[3:] - z[3:])


def main():
    config = DroneConfig()
    cbf = CBF.from_config(config)
    env = DroneEnv(config.pos_min, config.pos_max)

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
