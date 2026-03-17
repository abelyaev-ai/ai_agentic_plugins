---
name: rllib-environment-setup
description: >
  This skill should be used when the user asks to "create a custom RL environment",
  "scaffold a Gymnasium env", "define observation and action spaces",
  or sets up environments for RLlib.
---

## Purpose

Provide patterns and best practices for creating custom Gymnasium environments compatible with RLlib. This skill covers environment class scaffolding, observation and action space design, the Gymnasium 5-tuple return format, environment registration, vectorized environment support, and validation techniques.

## Prerequisites

Before implementing any environment, resolve the relevant RLlib library documentation via Context7:

1. Call `resolve-library-id` with `libraryName: "ray rllib"` and a query describing the environment requirements.
2. Call `query-docs` with the resolved library ID and specific questions about environment APIs, space types, or RLlib integration patterns.

Ground all implementations in the latest API surface rather than memorized patterns.

## Core Workflow

### Step 1: Environment Class Scaffold

Create a custom environment by subclassing `gymnasium.Env`. The constructor accepts a single `config` argument (defaulting to `None`) and defines both observation and action spaces.

```python
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class CustomEnv(gym.Env):
    """Custom environment following Gymnasium interface.

    Attributes:
        observation_space: Defines valid observations the environment produces.
        action_space: Defines valid actions the agent can take.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__()
        config = config or {}

        # Extract configuration parameters with defaults
        self._max_steps: int = config.get("max_steps", 100)
        self._step_count: int = 0

        # Define spaces in constructor (required by Gymnasium)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)
```

### Step 2: Observation Space Design

Select the appropriate space type based on the observation structure:

**Box** - Continuous n-dimensional arrays with bounded or unbounded values:

```python
# Bounded continuous observations
self.observation_space = gym.spaces.Box(
    low=np.array([0.0, -np.inf, -1.0], dtype=np.float32),
    high=np.array([10.0, np.inf, 1.0], dtype=np.float32),
    dtype=np.float32,
)

# Image observations (H, W, C format)
self.observation_space = gym.spaces.Box(
    low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
)
```

**Discrete** - Single categorical value from a finite set:

```python
# 5 possible states: 0, 1, 2, 3, 4
self.observation_space = gym.spaces.Discrete(5)
```

**MultiDiscrete** - Multiple independent categorical values:

```python
# Three features with 3, 5, and 2 categories respectively
self.observation_space = gym.spaces.MultiDiscrete([3, 5, 2])
```

**MultiBinary** - Binary feature vector of fixed length:

```python
# 10-dimensional binary observation
self.observation_space = gym.spaces.MultiBinary(10)
```

**Dict** - Composite observations with named components:

```python
self.observation_space = gym.spaces.Dict({
    "position": gym.spaces.Box(-10.0, 10.0, (3,), np.float32),
    "velocity": gym.spaces.Box(-5.0, 5.0, (3,), np.float32),
    "target": gym.spaces.Discrete(8),
})
```

**Tuple** - Composite observations with positional components:

```python
self.observation_space = gym.spaces.Tuple((
    gym.spaces.Box(-1.0, 1.0, (4,), np.float32),
    gym.spaces.Discrete(3),
))
```

### Step 3: Action Space Design

Action spaces follow the same type system as observation spaces. Common patterns include:

```python
# Discrete actions (e.g., move left/right/stay)
self.action_space = gym.spaces.Discrete(3)

# Continuous control (e.g., joint torques)
self.action_space = gym.spaces.Box(
    low=-1.0, high=1.0, shape=(4,), dtype=np.float32
)

# Hybrid action spaces using Dict
self.action_space = gym.spaces.Dict({
    "move": gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
    "jump": gym.spaces.Discrete(2),
})
```

### Step 4: Implement reset() Method

The `reset()` method initializes the environment state and returns the initial observation with an info dictionary. Include `seed` and `options` parameters for reproducibility and configuration.

```python
def reset(
    self,
    *,
    seed: Optional[int] = None,
    options: Optional[dict[str, Any]] = None,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Reset the environment to initial state.

    Args:
        seed: Random seed for reproducibility. Pass to super().reset().
        options: Additional reset options (environment-specific).

    Returns:
        Tuple of (initial_observation, info_dict).
    """
    super().reset(seed=seed)  # Initialize RNG if seed provided

    # Reset internal state
    self._step_count = 0
    self._state = self.np_random.uniform(-0.5, 0.5, size=(4,)).astype(np.float32)

    # Construct initial observation
    observation = self._get_observation()

    # Info dict can contain diagnostic information
    info: dict[str, Any] = {"initial_state": self._state.copy()}

    return observation, info
```

### Step 5: Implement step() Method

The `step()` method executes an action and returns the Gymnasium 5-tuple: `(observation, reward, terminated, truncated, info)`.

```python
def step(
    self, action: int
) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
    """Execute one environment step.

    Args:
        action: The action to execute (type matches action_space).

    Returns:
        Tuple of (observation, reward, terminated, truncated, info):
            observation: New state observation.
            reward: Scalar reward signal.
            terminated: True if episode ended due to terminal state.
            truncated: True if episode ended due to time limit.
            info: Diagnostic information dictionary.
    """
    assert self.action_space.contains(action), f"Invalid action: {action}"

    self._step_count += 1

    # Apply action and update state
    self._state = self._transition(self._state, action)

    # Compute reward
    reward = self._compute_reward(self._state, action)

    # Check termination conditions
    terminated = self._is_terminal(self._state)
    truncated = self._step_count >= self._max_steps

    # Build observation
    observation = self._get_observation()

    # Additional diagnostic info
    info: dict[str, Any] = {
        "step_count": self._step_count,
        "state": self._state.copy(),
    }

    return observation, reward, terminated, truncated, info
```

**Critical distinction between terminated and truncated:**

- `terminated=True`: Episode ended due to reaching a terminal state (goal achieved, agent died, game over). The final observation represents a true terminal state.
- `truncated=True`: Episode ended due to external limits (max steps, time limit). The final observation is NOT a true terminal state; the episode was cut short.

### Step 6: Environment Registration

Register the environment with Gymnasium to enable string-based instantiation:

```python
import gymnasium as gym
from my_package.envs.custom_env import CustomEnv

# Register with a unique ID
gym.register(
    id="MyCustomEnv-v0",
    entry_point="my_package.envs.custom_env:CustomEnv",
    max_episode_steps=1000,  # Auto-truncation after this many steps
    kwargs={"default_param": 42},  # Default config values
)

# Now instantiate via string
env = gym.make("MyCustomEnv-v0", max_steps=500)
```

For RLlib, pass the class directly or register it:

```python
from ray.rllib.algorithms.ppo import PPOConfig

# Direct class reference (preferred)
config = PPOConfig().environment(CustomEnv, env_config={"max_steps": 200})

# Or registered string
config = PPOConfig().environment("MyCustomEnv-v0")
```

### Step 7: Vectorized Environment Support

Enable parallel environment execution for faster data collection. Gymnasium supports synchronous and asynchronous vectorization modes.

```python
import gymnasium as gym

# Synchronous vectorization (simple, deterministic)
vec_env = gym.make_vec("MyCustomEnv-v0", num_envs=8, vectorization_mode="sync")

# Asynchronous vectorization (faster, uses multiprocessing)
vec_env = gym.make_vec("MyCustomEnv-v0", num_envs=8, vectorization_mode="async")
```

Configure vectorization in RLlib via `num_envs_per_env_runner`:

```python
config = (
    PPOConfig()
    .environment(CustomEnv, env_config={"max_steps": 200})
    .env_runners(
        num_env_runners=4,
        num_envs_per_env_runner=8,  # 32 total parallel envs
    )
)
```

### Step 8: Environment Validation

Validate environment correctness before training using `gymnasium.utils.env_checker.check_env`:

```python
from gymnasium.utils.env_checker import check_env

env = CustomEnv(config={"max_steps": 100})
check_env(env, skip_render_check=True)  # Raises on errors
```

Create comprehensive validation tests:

```python
import pytest
import numpy as np

def test_observation_space_contains_reset_observation():
    """Verify reset() returns observation within observation_space."""
    env = CustomEnv()
    obs, info = env.reset()
    assert env.observation_space.contains(obs), (
        f"Observation {obs} not in space {env.observation_space}"
    )

def test_observation_space_contains_step_observation():
    """Verify step() returns observation within observation_space."""
    env = CustomEnv()
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)

def test_episode_terminates():
    """Verify episodes eventually end."""
    env = CustomEnv(config={"max_steps": 10})
    env.reset()
    done = False
    steps = 0
    while not done and steps < 1000:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    assert done, "Episode did not terminate within expected steps"

def test_deterministic_reset_with_seed():
    """Verify seeded reset produces identical results."""
    env1 = CustomEnv()
    env2 = CustomEnv()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)
```

## Common Pitfalls

- **Returning wrong tuple length in step()**: Always return exactly 5 values `(obs, reward, terminated, truncated, info)`. Legacy 4-tuple `(obs, reward, done, info)` causes RLlib failures.

- **Confusing terminated and truncated**: Use `terminated=True` only for true terminal states (goal, death). Use `truncated=True` for artificial limits (max steps). Incorrect usage corrupts value function learning.

- **Observation dtype mismatch**: Ensure observations match the space dtype exactly. A `Box(..., dtype=np.float32)` space requires `np.float32` arrays, not `float64`.

- **Missing space bounds validation**: Always verify actions and observations stay within declared space bounds. Use `assert self.action_space.contains(action)` in `step()` during development.

- **Forgetting to call super().reset(seed=seed)**: The parent class initializes `self.np_random` when a seed is provided. Skipping this breaks reproducibility.

## Additional Resources

Refer to `references/env-patterns.md` for:

- MultiAgentEnv implementation patterns
- External environment connectors
- Environment wrapper patterns (ObservationWrapper, ActionWrapper, RewardWrapper)
- Observation normalization techniques
- Action masking implementation
- Environment validation and debugging strategies
- Testing fixtures for environment correctness
- RLlib environment configuration via AlgorithmConfig
