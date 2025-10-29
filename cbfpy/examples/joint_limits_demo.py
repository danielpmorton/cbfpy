"""
# Manipulator Joint Limit Avoidance Demo

We will command a joint position trajectory that exceeds the joint limits, and the CBF will ensure that we stay
within the limits (+ some margin)

This uses a single-integrator reduced model of the manipulator dynamics.
We define the state as the joint positions and assume that we can directly control the joint velocities
i.e. z = [q1, q2, q3] and u = [q1_dot, q2_dot, q3_dot]
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cbfpy import CBF, CBFConfig
from cbfpy.envs.arm_envs import JointLimitsEnv
from cbfpy.utils.general_utils import find_assets_dir

URDF = find_assets_dir() + "three_dof_arm.urdf"


class JointLimitsConfig(CBFConfig):
    """Config for the 3-DOF arm, avoiding its joint limits using velocity control"""

    def __init__(self):
        self.num_joints = 3
        # Joint limit values from the URDF
        self.q_min = -np.pi / 2 * np.ones(self.num_joints)
        self.q_max = np.pi / 2 * np.ones(self.num_joints)
        # Pad joint limts (to better evauate CBF performance)
        self.padding = 0.3
        super().__init__(n=self.num_joints, m=self.num_joints)

    def f(self, z):
        return jnp.zeros(self.num_joints)

    def g(self, z):
        return jnp.eye(self.num_joints)

    def h_1(self, z):
        q = z
        return jnp.concatenate(
            [self.q_max - q - self.padding, q - self.q_min - self.padding]
        )


def nominal_controller(q: Array, q_des: Array) -> Array:
    """Very simple proportional controller: Commands joint velocities to reduce a position error

    Args:
        q (Array): Joint positions, shape (num_joints,)
        q_des (Array): Desired joint positions, shape (num_joints,)

    Returns:
        Array: Joint velocity command, shape (num_joints,)
    """
    k = 1.0
    return k * (q_des - q)


def main():
    config = JointLimitsConfig()
    cbf = CBF.from_config(config)
    env = JointLimitsEnv()

    @jax.jit
    def safety_filter(q, u):
        return cbf.safety_filter(q, u)

    q_hist = []
    q_des_hist = []
    u_safe_hist = []
    u_unsafe_hist = []
    sim_time = 100.0
    timestep = env.timestep
    num_timesteps = int(sim_time / timestep)
    for i in range(num_timesteps):
        q = env.get_state()
        q_des = env.get_desired_state()
        u_unsafe = nominal_controller(q, q_des)
        u = safety_filter(q, u_unsafe)
        env.apply_control(u)
        env.step()
        q_hist.append(q)
        q_des_hist.append(q_des)
        u_unsafe_hist.append(u_unsafe)
        u_safe_hist.append(u)

    ## Plotting ##

    fig, axs = plt.subplots(2, 3)
    ts = timestep * np.arange(num_timesteps)

    # On the top row, plot the q and q des for each joint, along with the joint limits indicated
    for i in range(3):
        (q_line,) = axs[0, i].plot(ts, np.array(q_hist)[:, i], label="q")
        (q_des_line,) = axs[0, i].plot(ts, np.array(q_des_hist)[:, i], label="q_des")
        axs[0, i].plot(
            ts,
            env.q_min[i] * np.ones(num_timesteps),
            ls="--",
            c="red",
            label="q_min (True)",
        )
        axs[0, i].plot(
            ts,
            env.q_max[i] * np.ones(num_timesteps),
            ls="--",
            c="red",
            label="q_max (True)",
        )
        axs[0, i].plot(
            ts,
            (env.q_min[i] + config.padding) * np.ones(num_timesteps),
            ls="--",
            c="blue",
            label="q_min (CBF)",
        )
        axs[0, i].plot(
            ts,
            (env.q_max[i] - config.padding) * np.ones(num_timesteps),
            ls="--",
            c="blue",
            label="q_max (CBF)",
        )
        legend_elements = [
            q_line,
            q_des_line,
            Line2D([0], [0], color="red", ls="--", label="True limits"),
            Line2D([0], [0], color="blue", ls="--", label="CBF limits"),
        ]
        axs[0, i].legend(handles=legend_elements, loc="lower left")
        axs[0, i].set_title(f"Joint {i} position")

    # On the bottom row, plot the safe and unsafe velocities for each joint
    for i in range(3):
        axs[1, i].plot(ts, np.array(u_safe_hist)[:, i], label="Safe")
        axs[1, i].plot(ts, np.array(u_unsafe_hist)[:, i], label="Unsafe")
        axs[1, i].legend(loc="lower left")
        axs[1, i].set_title(f"Joint {i} velocity")

    plt.show()


if __name__ == "__main__":
    main()
