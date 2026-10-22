---
layout: "contents"
title: Create custom env
---

# Create a Custom Environment

## Before You Code: Environment Design

Creating an RL environment is like designing a video game or simulation. Before writing any code, you need to think through the learning problem you want to solve. This design phase is crucial - a poorly designed environment will make learning difficult or impossible, no matter how good your algorithm is.

### Key Design Questions

Ask yourself these fundamental questions:

**🎯 What skill should the agent learn?**
- Navigate through a maze?
- Balance and control a system?
- Optimize resource allocation?
- Play a strategic game?

**👀 What information does the agent need?**
- Position and velocity?
- Current state of the system?
- Historical data?
- Partial or full observability?

**🎮 What actions can the agent take?**
- Discrete choices (move up/down/left/right)?
- Continuous control (steering angle, throttle)?
- Multiple simultaneous actions?

**🏆 How do we measure success?**
- Reaching a specific goal?
- Minimizing time or energy?
- Maximizing a score?
- Avoiding failures?

**⏰ When should episodes end?**
- Task completion (success/failure)?
- Time limits?
- Safety constraints?

### GridWorld Example Design

For our tutorial example, we'll create a simple GridWorld environment:

- **🎯 Skill**: Navigate efficiently to a target location
- **👀 Information**: Agent position and target position on a grid
- **🎮 Actions**: Move up, down, left, or right
- **🏆 Success**: Reach the target in minimum steps
- **⏰ End**: When agent reaches target (or optional time limit)

This provides a clear learning problem that's simple enough to understand but non-trivial to solve optimally.

---

This page provides a complete implementation of creating custom environments with Gymnasium. For a longer tutorial that also covers rendering with PyGame, see the [environment creation tutorial](../tutorials/gymnasium_basics/environment_creation).

We recommend that you familiarise yourself with the [basic usage](basic_usage) before reading this page!

We will implement our GridWorld game as a 2-dimensional square grid of fixed size. The agent can move vertically or horizontally between grid cells in each timestep, and the goal is to navigate to a target that has been placed randomly at the beginning of the episode.

## Environment `__init__`

```{eval-rst}
.. py:currentmodule:: gymnasium

Like all environments, our custom environment will inherit from :class:`gymnasium.Env` that defines the structure all environments must follow. One of the requirements is defining the observation and action spaces, which declare what inputs (actions) and outputs (observations) are valid for this environment.

As outlined in our design, our agent has four discrete actions (move in cardinal directions), so we'll use ``Discrete(4)`` space.
```

```{eval-rst}
.. py:currentmodule:: gymnasium.spaces

For our observation, we have several options. We could represent the full grid as a 2D array, or use coordinate positions, or even a 3D array with separate "layers" for agent and target. For this tutorial, we'll use a simple dictionary format like ``{"agent": array([1, 0]), "target": array([0, 3])}`` where the arrays are ``[row, col]`` grid coordinates (row 0 is the top row, as in a NumPy array).

This choice makes the observation human-readable and easy to debug. We'll declare this as a :class:`Dict` space with the agent and target spaces being :class:`Box` spaces that contain integer coordinates.
```

For a full list of possible spaces to use with an environment, see [spaces](../api/spaces)

```python
from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        # The size of the square grid (5x5 by default)
        self.size = size

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [row, col] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [row, col] coordinates
            }
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([0, 1]),   # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),   # Move down (row + 1)
        }
```

## Constructing Observations

```{eval-rst}
.. py:currentmodule:: gymnasium

Since we need to compute observations in both :meth:`Env.reset` and :meth:`Env.step`, it's convenient to have a helper method ``_get_obs`` that translates the environment's internal state into the observation format. This keeps our code DRY (Don't Repeat Yourself) and makes it easier to modify the observation format later.
```

<!-- test: skip -->
```python
    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        # Return copies so that the observations handed to the user can't be
        # changed by later steps (or change the environment's internal state)
        return {"agent": self._agent_location.copy(), "target": self._target_location.copy()}
```

```{eval-rst}
.. py:currentmodule:: gymnasium

We can also implement a similar method for auxiliary information returned by :meth:`Env.reset` and :meth:`Env.step`. In our case, we'll provide the Manhattan distance between agent and target - this can be useful for debugging and understanding agent progress, but shouldn't be used by the learning algorithm itself.
```

<!-- test: skip -->
```python
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
```

```{eval-rst}
.. py:currentmodule:: gymnasium

Sometimes info will contain data that's only available inside :meth:`Env.step` (like individual reward components, action success/failure, etc.). In those cases, we'd update the dictionary returned by ``_get_info`` directly in the step method.
```

## Reset function

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

The :meth:`reset` method starts a new episode. It takes two optional parameters: ``seed`` for reproducible random generation and ``options`` for additional configuration. On the first line, you must call ``super().reset(seed=seed)`` to properly initialize the random number generator.

In our GridWorld environment, :meth:`reset` randomly places the agent and target on the grid, ensuring they don't start in the same location. We return both the initial observation and info as a tuple.
```

<!-- test: skip -->
```python
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
```

## Step function

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

The :meth:`step` method contains the core environment logic. It takes an action, updates the environment state, and returns the results. This is where the physics, game rules, and reward logic live.

For GridWorld, we need to:

1. Convert the discrete action to a movement direction
2. Update the agent's position (with boundary checking)
3. Calculate the reward based on whether the target was reached
4. Determine if the episode should end
5. Return all the required information
```

<!-- test: skip -->
```python
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
```

## The Complete Environment

Putting the pieces together, here is the full environment in one place, ready to copy:

```python
from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        self.size = size
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [row, col] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [row, col] coordinates
            }
        )
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 1]),   # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),   # Move down (row + 1)
        }

    def _get_obs(self):
        return {"agent": self._agent_location.copy(), "target": self._target_location.copy()}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        return self._get_obs(), self._get_info()

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0

        return self._get_obs(), reward, terminated, truncated, self._get_info()
```


## Common Environment Design Pitfalls

Now that you've seen the basic structure, let's discuss common mistakes beginners make:

### Reward Design Issues

**Problem**: Only rewarding at the very end (sparse rewards)
<!-- test: skip -->
```python
# Fine for our small 5x5 grid, but in larger or harder environments
# a random agent may rarely see any reward, making learning slow
reward = 1 if terminated else 0
```

**Often better**: Provide intermediate feedback
<!-- test: skip -->
```python
# Option 1: Small step penalty to encourage efficiency
reward = 1 if terminated else -0.01

# Option 2: Distance-based reward shaping (Manhattan distance, as in _get_info)
distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)
reward = 1 if terminated else -0.1 * distance
```

### State Representation Problems

**Problem**: Including irrelevant information or missing crucial details
<!-- test: skip -->
```python
# Too much info - agent doesn't need grid size in every observation
obs = {"agent": self._agent_location, "target": self._target_location, "size": self.size}

# Too little info - agent can't distinguish different positions
obs = {"distance": distance}  # Missing actual positions!
```

**Better**: Include exactly what's needed for optimal decisions
<!-- test: skip -->
```python
# Just right - positions are sufficient for navigation
obs = {"agent": self._agent_location, "target": self._target_location}
```

### Action Space Issues

**Problem**: Actions that don't make sense or are impossible to execute
<!-- test: skip -->
```python
# Bad: Agent can move diagonally but environment doesn't support it
self.action_space = gym.spaces.Discrete(8)  # 8 directions including diagonals

# Bad: Continuous actions for discrete movement
self.action_space = gym.spaces.Box(-1, 1, shape=(2,))  # Continuous x,y movement
```

### Boundary Handling Errors

**Problem**: Allowing invalid states or unclear boundary behavior
<!-- test: skip -->
```python
# Bad: Agent can go outside the grid
self._agent_location = self._agent_location + direction  # No bounds checking!

# Unclear: What happens when agent hits wall?
if np.any(self._agent_location < 0) or np.any(self._agent_location >= self.size):
    # Do nothing? Reset episode? Give penalty? Unclear!
```

**Better**: Clear, consistent boundary handling
<!-- test: skip -->
```python
# Clear: Agent stays in place when hitting boundaries
self._agent_location = np.clip(
    self._agent_location + direction, 0, self.size - 1
)
```

## Registering and making the environment

```{eval-rst}
While you can use your custom environment immediately, it's more convenient to register it with Gymnasium so you can create it with :meth:`gymnasium.make` just like built-in environments.

The environment ID has three components: an optional namespace (here: ``gymnasium_env``), a mandatory name (here: ``GridWorld``), and an optional but recommended version (here: v0). You could register it as ``GridWorld-v0``, ``GridWorld``, or ``gymnasium_env/GridWorld``, but the full format is recommended for clarity.

Since this tutorial isn't part of a Python package, we pass the class directly as the entry point. In real projects, you'd typically use a string like ``"my_package.envs:GridWorldEnv"``.
```

```python
# Register the environment so we can create it with gym.make()
gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
    max_episode_steps=300,  # Prevent infinite episodes
)
```

For a more complete guide on registering custom environments (including with string entry points), please read the full [create environment](../tutorials/gymnasium_basics/environment_creation) tutorial.

```{eval-rst}
Once registered, you can check all available environments with :meth:`gymnasium.pprint_registry` and create instances with :meth:`gymnasium.make`. You can also create vectorized versions with :meth:`gymnasium.make_vec`.
```

```python
>>> import gymnasium as gym

>>> # Create the environment like any built-in environment
>>> env = gym.make("gymnasium_env/GridWorld-v0")
>>> env
<TimeLimit<OrderEnforcing<PassiveEnvChecker<GridWorldEnv<gymnasium_env/GridWorld-v0>>>>>

>>> # Customize environment parameters
>>> env = gym.make("gymnasium_env/GridWorld-v0", size=10)
>>> env.unwrapped.size
10

>>> # Create multiple environments for parallel training
>>> vec_env = gym.make_vec("gymnasium_env/GridWorld-v0", num_envs=3)
>>> vec_env
SyncVectorEnv(gymnasium_env/GridWorld-v0, num_envs=3)
```


## Debugging Your Environment

When your environment doesn't work as expected, here are common debugging strategies:

### Check Environment Validity
```python
from gymnasium.utils.env_checker import check_env

env = gym.make("gymnasium_env/GridWorld-v0")

# This will catch many common issues.
# Check the unwrapped environment, as wrappers (like TimeLimit) can hide problems
try:
    check_env(env.unwrapped)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
```

### Manual Testing with Known Actions
```python
# Test specific action sequences to verify behavior
env = gym.make("gymnasium_env/GridWorld-v0")
obs, info = env.reset(seed=42)  # Use seed for reproducible testing

print(f"Starting position - Agent: {obs['agent']}, Target: {obs['target']}")

# Test each action type
actions = [0, 1, 2, 3]  # right, up, left, down
for action in actions:
    old_pos = obs['agent']
    obs, reward, terminated, truncated, info = env.step(action)
    new_pos = obs['agent']
    print(f"Action {action}: {old_pos} -> {new_pos}, reward={reward}")
```

Positions are ``[row, col]``, so "up" decreases the row, and moving into a wall leaves the agent where it is:

```
Starting position - Agent: [0 3], Target: [3 2]
Action 0: [0 3] -> [0 4], reward=0
Action 1: [0 4] -> [0 4], reward=0
Action 2: [0 4] -> [0 3], reward=0
Action 3: [0 3] -> [1 3], reward=0
```

### Common Debug Issues
<!-- test: skip -->
```python
# Issue 1: Forgot to call super().reset()
def reset(self, seed=None, options=None):
    # super().reset(seed=seed)  # ❌ Missing this line
    # Results in: possibly incorrect seeding

# Issue 2: Mixing up coordinate conventions
# Our positions are NumPy-style [row, col], with row 0 at the top of the screen.
# Writing directions as Cartesian [x, y] instead moves the agent the wrong way:
self._action_to_direction = {
    0: np.array([1, 0]),   # ❌ Intended as "right" but this changes the row (moves down)
    1: np.array([0, 1]),   # ❌ Intended as "up" but this changes the column (moves right)
}

# Issue 3: Not handling boundaries properly
# This allows agent to go outside the grid!
self._agent_location = self._agent_location + direction  # ❌ No bounds checking
```

## Using Wrappers

Sometimes you want to modify your environment's behavior without changing the core implementation. Wrappers are perfect for this - they let you add functionality like changing observation formats, adding time limits, or modifying rewards without touching your original environment code.

```python
>>> from gymnasium.wrappers import FlattenObservation

>>> # Original observation is a dictionary
>>> env = gym.make('gymnasium_env/GridWorld-v0')
>>> env.observation_space
Dict('agent': Box(0, 4, (2,), int64), 'target': Box(0, 4, (2,), int64))

>>> obs, info = env.reset(seed=42)
>>> obs
{'agent': array([0, 3]), 'target': array([3, 2])}

>>> # Wrap it to flatten observations into a single array
>>> wrapped_env = FlattenObservation(env)
>>> wrapped_env.observation_space
Box(0, 4, (4,), int64)

>>> obs, info = wrapped_env.reset(seed=42)
>>> obs  # [agent_row, agent_col, target_row, target_col]
array([0, 3, 3, 2])
```

This is particularly useful when working with algorithms that expect specific input formats (like neural networks that need 1D arrays instead of dictionaries).

## Advanced Environment Features

Once you have the basics working, you might want to add more sophisticated features:

### Adding Rendering

Declare the supported render modes in ``metadata`` and store the ``render_mode`` passed to ``__init__`` (``gym.make(..., render_mode="ansi")`` forwards it for you):

<!-- test: skip -->
```python
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, size: int = 5, render_mode: Optional[str] = None):
        self.size = size
        self.render_mode = render_mode
        # ... rest of init ...

    def render(self):
        """Render the environment as text."""
        if self.render_mode == "ansi":
            # Build a simple ASCII grid, row 0 at the top
            grid = ""
            for row in range(self.size):
                for col in range(self.size):
                    if np.array_equal([row, col], self._agent_location):
                        grid += "A "  # Agent
                    elif np.array_equal([row, col], self._target_location):
                        grid += "T "  # Target
                    else:
                        grid += ". "  # Empty
                grid += "\n"
            return grid
```

With this added, `env = gym.make("gymnasium_env/GridWorld-v0", render_mode="ansi")`, `env.reset(seed=42)` and `print(env.render())` show the agent (`A`) and target (`T`) from the seeded reset above:

```text
. . . A .
. . . . .
. . . . .
. . T . .
. . . . .
```

### Parameterized Environments
<!-- test: skip -->
```python
def __init__(self, size: int = 5, reward_scale: float = 1.0, step_penalty: float = 0.0):
    self.size = size
    self.reward_scale = reward_scale
    self.step_penalty = step_penalty
    # ... rest of init ...

def step(self, action):
    # ... movement logic ...

    # Flexible reward calculation
    if terminated:
        reward = self.reward_scale  # Success reward
    else:
        reward = -self.step_penalty  # Step penalty (0 by default)
```

## Real-World Environment Design Tips

### Start Simple, Add Complexity Gradually
1. **First**: Get basic movement and goal-reaching working
2. **Then**: Add obstacles, multiple goals, or time pressure
3. **Finally**: Add complex dynamics, partial observability, or multi-agent interactions

### Design for Learning
- **Clear Success Criteria**: Agent should know when it's doing well
- **Reasonable Difficulty**: Not too easy (trivial) or too hard (impossible)
- **Consistent Rules**: Same action in same state should have same effect
- **Informative Observations**: Include everything needed for optimal decisions

### Think About Your Research Question
- **Navigation**: Focus on spatial reasoning and path planning
- **Control**: Emphasize dynamics, stability, and continuous actions
- **Strategy**: Include partial information, opponent modeling, or long-term planning
- **Optimization**: Design clear trade-offs and resource constraints

## Next Steps

Congratulations! You now know how to create custom RL environments. Here's what to explore next:

1. **Add rendering** to visualize your environment ([complete tutorial](../tutorials/gymnasium_basics/environment_creation))
2. **Train an agent** on your custom environment ([training guide](train_agent))
3. **Experiment with different reward functions** to see how they affect learning
4. **Try wrapper combinations** to modify your environment's behavior
5. **Create more complex environments** with obstacles, multiple agents, or continuous actions

The key to good environment design is iteration - start simple, test thoroughly, and gradually add complexity as needed for your research or application goals.
