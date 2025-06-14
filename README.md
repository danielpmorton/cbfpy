<div align="center">
<img src="https://github.com/user-attachments/assets/92c36a77-40d9-44d8-afa2-12417941534c" alt="logo"></img>
</div>

# CBFpy: Control Barrier Functions in Python and Jax

CBFpy is an easy-to-use and high-performance framework for constructing and solving Control Barrier Functions (CBFs) and Control Lyapunov Functions (CLFs), using [Jax](https://github.com/google/jax) for:

- Just-in-time compilation
- Accelerated linear algebra operations with [XLA](https://openxla.org/xla)
- Automatic differentiation

For API reference, see the following [documentation](https://danielpmorton.github.io/cbfpy)

If you use CBFpy in your research, please cite the following [paper](https://arxiv.org/abs/2503.06736):

```
@article{morton2025oscbf,
  author = {Morton, Daniel and Pavone, Marco},
  title = {Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions},
  year = {2025},
  journal = {arXiv preprint arXiv:2503.06736},
  note = {Submitted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Hangzhou, 2025},
}
```

## Installation 

### From PyPI

```
pip install cbfpy
```

### From source

A virtual environment is optional, but highly recommended. For `pyenv` installation instructions, see [here](https://danielpmorton.github.io/cbfpy/pyenv).

```
git clone https://github.com/danielpmorton/cbfpy
cd cbfpy
pip install -e ".[examples]"
```
The `[examples]` tag installs all of the required packages for development and running the examples. The pure `cbfpy` functionality does not require these extra packages though. If you want to contribute to the repo, you can also include the `[dev]` dependencies.

If you are working on Apple silicon and have issues installing Jax, the following threads may be useful: [[1]](https://stackoverflow.com/questions/68327863/importing-jax-fails-on-mac-with-m1-chip), [[2]](https://github.com/jax-ml/jax/issues/5501#issuecomment-955590288)

## Usage:

#### Example: A point-mass robot in 1D with an applied force and a positional barrier

For this problem, the state $z$ is defined as the position and velocity of the robot,

$$z = [x, \dot{x}]$$ 

So, the state derivative $\dot{z}$ is therefore

$$\dot{z} = [\dot{x}, \ddot{x}]$$ 

And the control input is the applied force in the $x$ direction:

$$u = F_{x}$$

The dynamics can be expressed as follows (with $m$ denoting the robot's mass):

$$\dot{z} = \begin{bmatrix}0 & 1 \\
                           0 & 0
            \end{bmatrix}z + 
            \begin{bmatrix}0 \\
                          1/m
            \end{bmatrix} u$$

This is a control affine system, since the dynamics can be expressed as 

$$\dot{z} = f(z) + g(z) u$$

If the robot is controlled by some nominal (unsafe) controller, we may want to guarantee that it remains in some safe region. If we define $X_{safe} \in [x_{min}, \infty]$, we can construct a (relative-degree-2, zeroing) barrier $h$ where $h(z) \geq 0$ for any $z$ in the safe set:

$$h(z) = x - x_{min}$$

### In Code

We'll first define our problem (dynamics, barrier, and any additional parameters) in a `CBFConfig`-derived class. 

We use [Jax](https://github.com/google/jax) for fast compilation of the problem. Jax can be tricky to learn at first, but luckily `cbfpy` just requires formulating your functions in `jax.numpy` which has the same familiar interface as `numpy`. These should be pure functions without side effects (for instance, modifying a class variable in `self`).

Additional tuning parameters/functions can be found in the `CBFConfig` documentation. 

```python
import jax.numpy as jnp
from cbfpy import CBF, CBFConfig

# Create a config class for your problem inheriting from the CBFConfig class
class MyCBFConfig(CBFConfig):
    def __init__(self):
        super().__init__(
            # Define the state and control dimensions
            n = 2, # [x, x_dot]
            m = 1, # [F_x]
            # Define control limits (if desired)
            u_min = None,
            u_max = None,
        )

    # Define the control-affine dynamics functions `f` and `g` for your system
    def f(self, z):
        A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        return A @ z

    def g(self, z):
        mass = 1.0
        B = jnp.array([[0.0], [1.0 / mass]])
        return B

    # Define the barrier function `h`
    # The *relative degree* of this system is 2, so, we'll use the h_2 method
    def h_2(self, z):
        x_min = 1.0
        x = z[0]
        return jnp.array([x - x_min])
```
We can then construct the CBF from our config and use it in our control loop as follows.
```python
config = MyCBFConfig()
cbf = CBF.from_config(config)

# Pseudocode
while True:
    z = get_state()
    z_des = get_desired_state()
    u_nom = nominal_controller(z, z_des)
    u = cbf.safety_filter(z, u_nom)
    apply_control(u)
    step() 
```

## Examples

These can be found in the `examples` folder [here](https://github.com/danielpmorton/cbfpy/tree/main/cbfpy/examples)

### [Adaptive Cruise Control](https://github.com/danielpmorton/cbfpy/blob/main/cbfpy/examples/adaptive_cruise_control_demo.py)

Use a CLF-CBF to maintain a safe follow distance to the vehicle in front, while tracking a desired velocity

- State: z = [Follower velocity, Leader velocity, Follow distance] (n = 3)
- Control: u = [Follower wheel force] (m = 1)
- Relative degree: 1

![Image: Adaptive cruise control](https://raw.githubusercontent.com/danielpmorton/cbfpy/refs/heads/main/images/acc_safe.gif)

### [Point Robot Safe-Set Containment](https://github.com/danielpmorton/cbfpy/blob/main/cbfpy/examples/point_robot_demo.py)

Use a CBF to enforce that a point robot stays within a safe box, while a PD controller attempts to reduce the distance to a target position

- State: z = [Position, Velocity] (n = 6)
- Control: u = [Force] (m = 3)
- Relative degree: 2

![Image: Point robot in a safe set](https://raw.githubusercontent.com/danielpmorton/cbfpy/refs/heads/main/images/point_robot_safe.gif)

### [Point Robot Obstacle Avoidance](https://github.com/danielpmorton/cbfpy/blob/main/cbfpy/examples/point_robot_obstacle_demo.py)

Use a CBF to keep a point robot inside a safe box, while avoiding a moving obstacle. The nominal PD controller attempts to keep the robot at the origin.

- State: z = [Position, Velocity] (n = 6)
- Control: u = [Force] (m = 3)
- Relative degree: 1 + 2 (1 for obstacle avoidance, 2 for safe set containment)
- Additional data: The state of the obstacle (position and velocity)

![Image: Point robot avoiding an obstacle](https://raw.githubusercontent.com/danielpmorton/cbfpy/refs/heads/main/images/point_robot_obstacle.gif)

### [Manipulator Joint Limit Avoidance](https://github.com/danielpmorton/cbfpy/blob/main/cbfpy/examples/joint_limits_demo.py)

Use a CBF to keep a manipulator operating within its joint limits, even if a nominal joint trajectory is unsafe. 

- State: z = [Joint angles] (n = 3)
- Control: u = [Joint velocities] (m = 3)
- Relative degree: 1

![Image: 3-DOF manipulator avoiding joint limits](https://raw.githubusercontent.com/danielpmorton/cbfpy/refs/heads/main/images/joint_limits.png)

### [Drone Obstacle Avoidance](https://github.com/danielpmorton/cbfpy/blob/main/cbfpy/examples/drone_demo.py)

Use a CBF to keep a drone inside a safe box, while avoiding a moving obstacle. This is similar to the "point robot obstacle avoidance" demo, but with slightly different dynamics.

- State: z = [Position, Velocity] (n = 6)
- Control: u = [Velocity] (m = 3)
- Relative degree: 1
- Additional data: The state of the obstacle (position and velocity)

This is the same CBF which was used in the ["Drone Fencing" demo](https://danielpmorton.github.io/drone_fencing/) at the Stanford Robotics center.

![Image: Quadrotor avoiding an obstacle](https://raw.githubusercontent.com/danielpmorton/cbfpy/refs/heads/main/images/drone_demo.gif)
