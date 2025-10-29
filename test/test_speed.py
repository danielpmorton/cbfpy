"""Speed tests for the CBF solver

We evaluate the speed of the solver NOT just via the QP solve but via the whole process
(solving for the nominal control input, constructing the QP matrices, and then solving).
This provides a more accurate view of what the controller frequency would actually be if
deployed on the robot.

These test cases can also be used to check that modifications to the CBF implementation
do not significantly degrade performance
"""

import unittest
from typing import Callable
import time
import jax
import numpy as np
import matplotlib.pyplot as plt

from cbfpy import CBF, CLFCBF
import cbfpy.examples.point_robot_demo as prdemo
import cbfpy.examples.adaptive_cruise_control_demo as accdemo

# Seed RNG for repeatability
np.random.seed(0)


# TODO Make this work if there are additional arguments for the barrier function
def eval_speed(
    controller_func: Callable,
    states: np.ndarray,
    des_states: np.ndarray,
    verbose: bool = True,
    plot: bool = True,
) -> float:
    """Tests the speed of a controller function via evaluation on a set of inputs

    Timing details (average solve time / Hz, distributions of times, etc.) can be printed to the terminal
    or visualized in plots, via the `verbose` and `plot` inputs

    Args:
        controller_func (Callable): Function to time. This should be the highest-level CBF-based controller
            function which includes the nominal controller, QP construction, and QP solve
        states (np.ndarray): Set of states to evaluate on, shape (num_evals, state_dim)
        des_states (np.ndarray): Set of desired states to evaluate on, shape (num_evals, des_state_dim)
        verbose (bool, optional): Whether to print timing details to the terminal. Defaults to True.
        plot (bool, optional): Whether to visualize the distribution of solve times. Defaults to True.

    Returns:
        float: Average solver Hz
    """
    assert isinstance(controller_func, Callable)
    assert isinstance(states, np.ndarray)
    assert isinstance(des_states, np.ndarray)
    assert isinstance(verbose, bool)
    assert isinstance(plot, bool)
    assert states.shape[0] > 1
    assert states.shape[0] == des_states.shape[0]
    controller_func: Callable = jax.jit(controller_func)

    # Do an initial solve to jit-compile the function
    start_time = time.perf_counter()
    u = controller_func(states[0], des_states[0])
    first_solve_time = time.perf_counter() - start_time

    # Solve for the remainder of the controls using the jit-compiled controller
    times = []
    for i in range(1, states.shape[0]):
        start_time = time.perf_counter()
        u = controller_func(states[i], des_states[i]).block_until_ready()
        times.append(time.perf_counter() - start_time)
    times = np.asarray(times)
    avg_solve_time = np.mean(times)
    max_solve_time = np.max(times)
    avg_hz = 1 / avg_solve_time
    worst_case_hz = 1 / max_solve_time

    if verbose:
        # Print info about solver stats
        print(f"Solved for the first control input in {first_solve_time} seconds")
        print(f"Average solve time: {avg_solve_time} seconds")
        print(f"Average Hz: {avg_hz}")
        print(f"Worst-case solve time: {max_solve_time}")
        print(f"Worst-case Hz: {worst_case_hz}")
        print(
            "NOTE: Worst-case behavior might be inaccurate due to how the OS manages background processes"
        )

    if plot:
        # Create a plot to visualize the distribution of times
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].hist(times, 20)
        axs[0, 0].set_title("Solve Times")
        axs[0, 0].set_ylabel("Frequency")
        axs[0, 0].set_xscale("log")
        axs[0, 1].boxplot(times, vert=False)
        axs[0, 1].set_title("Solve Times")
        axs[0, 1].set_xscale("log")
        axs[1, 0].hist(1 / times, 20)
        axs[1, 0].set_title("Hz")
        axs[1, 0].set_ylabel("Frequency")
        axs[1, 1].boxplot(1 / times, vert=False)
        axs[1, 1].set_title("Hz")
        plt.show()

    return avg_hz


@jax.tree_util.register_static
class PointRobotTest:
    """Test the speed of the point-robot-in-a-box demo, using randomly sampled states"""

    def __init__(self):
        self.config = prdemo.PointRobotConfig()
        self.cbf = CBF.from_config(self.config)
        self.nominal_controller = prdemo.nominal_controller
        self.pos_min = self.config.pos_min
        self.pos_max = self.config.pos_max

    def sample_states(self, num_samples: int) -> np.ndarray:
        """Sample a set of random states for the point robot (3D positions and velocities)"""
        # Sample positions uniformly inside the keep-in region
        positions = np.asarray(self.pos_min) + np.random.rand(
            num_samples, 3
        ) * np.subtract(self.pos_max, self.pos_min)
        # Assume x/y/z velocities are sampled uniformly between [-3, 3]
        velocities = -3.0 + 6 * np.random.rand(num_samples, 3)
        return np.column_stack([positions, velocities])

    @jax.jit
    def controller(self, z, z_des):
        u = self.nominal_controller(z, z_des)
        return self.cbf.safety_filter(z, u)

    def test_speed(self, verbose: bool = True, plot: bool = True):
        n_samples = 10000
        states = self.sample_states(n_samples)
        desired_states = self.sample_states(n_samples)
        avg_hz = eval_speed(self.controller, states, desired_states, verbose, plot)
        return avg_hz


@jax.tree_util.register_static
class ACCTest:
    """Test the speed of the adaptive cruise control CLF-CBF demo, using randomly sampled states"""

    def __init__(self):
        self.config = accdemo.ACCConfig()
        self.clf_cbf = CLFCBF.from_config(self.config)

    def sample_states(self, num_samples: int) -> np.ndarray:
        """Sample a set of random states for the adaptive cruise control demo"""
        follower_vels = np.random.rand(num_samples) * 20
        leader_vels = np.random.rand(num_samples) * 40
        distances = 10 + np.random.rand(num_samples) * 100
        return np.column_stack([follower_vels, leader_vels, distances])

    @jax.jit
    def controller(self, z, z_des):
        return self.clf_cbf.controller(z, z_des)

    def test_speed(self, verbose: bool = True, plot: bool = True):
        n_samples = 10000
        states = self.sample_states(n_samples)
        # Note that the desired states aren't really relevant for this specific demo
        # just due to how the ACC problem was constructed
        desired_states = self.sample_states(n_samples)
        avg_hz = eval_speed(self.controller, states, desired_states, verbose, plot)
        return avg_hz


class SpeedTest(unittest.TestCase):
    """Test case to guarantee that the CBFs run at least at kilohertz rates"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.point_robot_test = PointRobotTest()
        cls.acc_test = ACCTest()

    def test_point_robot(self):
        avg_hz = self.point_robot_test.test_speed(verbose=False, plot=False)
        print("Point robot average Hz: ", avg_hz)
        self.assertTrue(avg_hz >= 1000)

    def test_acc(self):
        avg_hz = self.acc_test.test_speed(verbose=False, plot=False)
        print("ACC average Hz: ", avg_hz)
        self.assertTrue(avg_hz >= 1000)


if __name__ == "__main__":
    unittest.main()
