# RLlib Environment Patterns Reference

This reference provides detailed patterns for advanced environment design, wrappers, validation, and RLlib integration.

## MultiAgentEnv Patterns

RLlib's `MultiAgentEnv` extends Gymnasium environments to support multiple agents with independent or shared policies.

### Basic MultiAgentEnv Structure

```python
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class TwoPlayerGame(MultiAgentEnv):
    """A two-player competitive environment.

    Each agent has its own observation and action space. Observations
    are returned as dictionaries keyed by agent ID.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__()
        config = config or {}

        # Define agent identifiers
        self.agents = self.possible_agents = ["player_1", "player_2"]

        # Per-agent observation spaces
        self.observation_spaces = {
            "player_1": gym.spaces.Box(-1.0, 1.0, (10,), np.float32),
            "player_2": gym.spaces.Box(-1.0, 1.0, (10,), np.float32),
        }

        # Per-agent action spaces
        self.action_spaces = {
            "player_1": gym.spaces.Discrete(4),
            "player_2": gym.spaces.Discrete(4),
        }

        self._board_state: Optional[NDArray[np.float32]] = None
        self._current_player: str = "player_1"

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        """Reset the environment.

        Returns:
            Tuple of (observations_dict, infos_dict) where keys are agent IDs.
        """
        super().reset(seed=seed)

        self._board_state = np.zeros(10, dtype=np.float32)
        self._current_player = "player_1"

        # Return observations for all agents that should act
        observations = {
            self._current_player: self._get_observation(self._current_player)
        }
        infos: dict[str, Any] = {self._current_player: {}}

        return observations, infos

    def step(
        self, action_dict: dict[str, int]
    ) -> tuple[
        dict[str, NDArray[np.float32]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Execute actions for all agents in action_dict.

        Args:
            action_dict: Mapping from agent_id to action.

        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos).
            Each is a dict keyed by agent_id, plus "__all__" for global done.
        """
        # Process actions from acting agents
        for agent_id, action in action_dict.items():
            self._apply_action(agent_id, action)

        # Compute next observations, rewards, termination
        observations: dict[str, NDArray[np.float32]] = {}
        rewards: dict[str, float] = {}
        terminateds: dict[str, bool] = {}
        truncateds: dict[str, bool] = {}
        infos: dict[str, Any] = {}

        game_over = self._check_game_over()

        for agent_id in self.agents:
            observations[agent_id] = self._get_observation(agent_id)
            rewards[agent_id] = self._compute_reward(agent_id)
            terminateds[agent_id] = game_over
            truncateds[agent_id] = False
            infos[agent_id] = {}

        # __all__ key indicates if episode is done for all agents
        terminateds["__all__"] = game_over
        truncateds["__all__"] = False

        return observations, rewards, terminateds, truncateds, infos

    def _get_observation(self, agent_id: str) -> NDArray[np.float32]:
        """Get observation for a specific agent."""
        assert self._board_state is not None
        # Agents may see the board from different perspectives
        if agent_id == "player_1":
            return self._board_state.copy()
        return -self._board_state.copy()  # Inverted view for player 2

    def _apply_action(self, agent_id: str, action: int) -> None:
        """Apply an agent's action to the environment."""
        pass  # Implementation specific to game rules

    def _compute_reward(self, agent_id: str) -> float:
        """Compute reward for a specific agent."""
        return 0.0  # Implementation specific

    def _check_game_over(self) -> bool:
        """Check if the game has ended."""
        return False  # Implementation specific
```

### Turn-Based Multi-Agent Environments

For turn-based games, return observations only for the currently acting agent:

```python
def step(
    self, action_dict: dict[str, int]
) -> tuple[
    dict[str, NDArray[np.float32]],
    dict[str, float],
    dict[str, bool],
    dict[str, bool],
    dict[str, Any],
]:
    """Execute turn-based step."""
    current_agent = self._current_player
    action = action_dict[current_agent]

    # Apply action
    self._apply_action(current_agent, action)

    # Switch to next player
    self._current_player = self._get_next_player()

    # Only return observation for the next acting agent
    observations = {
        self._current_player: self._get_observation(self._current_player)
    }
    rewards = {current_agent: self._compute_reward(current_agent)}
    terminateds = {current_agent: False, "__all__": self._check_game_over()}
    truncateds = {current_agent: False, "__all__": False}
    infos = {current_agent: {}}

    return observations, rewards, terminateds, truncateds, infos
```

### Parameter Sharing Across Agents

When agents share the same policy, ensure observation and action spaces are identical:

```python
class HomogeneousMultiAgentEnv(MultiAgentEnv):
    """Environment where all agents share the same spaces."""

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__()
        config = config or {}
        num_agents = config.get("num_agents", 4)

        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.possible_agents = self.agents.copy()

        # Shared spaces enable parameter sharing
        shared_obs_space = gym.spaces.Box(-1.0, 1.0, (20,), np.float32)
        shared_action_space = gym.spaces.Discrete(5)

        self.observation_spaces = {aid: shared_obs_space for aid in self.agents}
        self.action_spaces = {aid: shared_action_space for aid in self.agents}
```

Configure RLlib for parameter sharing:

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(HomogeneousMultiAgentEnv, env_config={"num_agents": 4})
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
    )
)
```

## External Environment Connectors

Connect RLlib to external simulators, games, or real-world systems.

### Async External Environment Pattern

```python
from __future__ import annotations

import queue
import threading
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class ExternalSimulatorEnv(gym.Env):
    """Connect to an external simulator via message queues.

    The simulator runs in a separate process/thread and communicates
    via observation and action queues.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__()
        config = config or {}

        self.observation_space = gym.spaces.Box(-10.0, 10.0, (8,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

        # Communication queues
        self._obs_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._action_queue: queue.Queue[NDArray[np.float32]] = queue.Queue()

        # Start external simulator connection
        self._simulator_thread = threading.Thread(
            target=self._run_simulator,
            args=(config.get("simulator_address", "localhost:8080"),),
            daemon=True,
        )
        self._simulator_thread.start()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Request reset from external simulator."""
        # Send reset command (implementation specific)
        self._send_reset_command(seed)

        # Wait for initial observation
        result = self._obs_queue.get(timeout=30.0)
        observation = np.array(result["observation"], dtype=np.float32)
        info = result.get("info", {})

        return observation, info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Send action to simulator and receive result."""
        # Send action
        self._action_queue.put(action)

        # Wait for step result
        result = self._obs_queue.get(timeout=30.0)

        observation = np.array(result["observation"], dtype=np.float32)
        reward = float(result["reward"])
        terminated = bool(result.get("terminated", False))
        truncated = bool(result.get("truncated", False))
        info = result.get("info", {})

        return observation, reward, terminated, truncated, info

    def _run_simulator(self, address: str) -> None:
        """Background thread managing simulator connection."""
        pass  # Implementation connects to external process

    def _send_reset_command(self, seed: Optional[int]) -> None:
        """Send reset command to simulator."""
        pass  # Implementation specific

    def close(self) -> None:
        """Clean up simulator connection."""
        # Signal shutdown and wait for thread
        pass  # Implementation specific
```

## Environment Wrapper Patterns

Gymnasium wrappers modify environment behavior without changing the base implementation.

### ObservationWrapper

Transform observations while preserving the underlying environment:

```python
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to zero mean and unit variance.

    Maintains running statistics for online normalization.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8) -> None:
        super().__init__(env)
        self._epsilon = epsilon

        # Running statistics
        obs_shape = env.observation_space.shape
        assert obs_shape is not None
        self._mean = np.zeros(obs_shape, dtype=np.float64)
        self._var = np.ones(obs_shape, dtype=np.float64)
        self._count = 0

        # Update observation space to normalized range
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )

    def observation(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize a single observation."""
        self._update_statistics(observation)
        normalized = (observation - self._mean) / np.sqrt(self._var + self._epsilon)
        return normalized.astype(np.float32)

    def _update_statistics(self, observation: NDArray[np.float32]) -> None:
        """Update running mean and variance."""
        self._count += 1
        delta = observation - self._mean
        self._mean += delta / self._count
        delta2 = observation - self._mean
        self._var += (delta * delta2 - self._var) / self._count


class FrameStack(gym.ObservationWrapper):
    """Stack consecutive frames as observation."""

    def __init__(self, env: gym.Env, num_frames: int = 4) -> None:
        super().__init__(env)
        self._num_frames = num_frames
        self._frames: list[NDArray[np.float32]] = []

        # Update observation space shape
        old_space = env.observation_space
        assert isinstance(old_space, gym.spaces.Box)
        low = np.repeat(old_space.low[np.newaxis, ...], num_frames, axis=0)
        high = np.repeat(old_space.high[np.newaxis, ...], num_frames, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=old_space.dtype
        )

    def reset(self, **kwargs: Any) -> tuple[NDArray[np.float32], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._frames = [obs] * self._num_frames
        return self.observation(obs), info

    def observation(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        self._frames.pop(0)
        self._frames.append(observation)
        return np.stack(self._frames, axis=0)
```

### ActionWrapper

Transform actions before passing to the base environment:

```python
class ClipAction(gym.ActionWrapper):
    """Clip continuous actions to action space bounds."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)

    def action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        """Clip action to valid range."""
        assert isinstance(self.action_space, gym.spaces.Box)
        return np.clip(action, self.action_space.low, self.action_space.high)


class DiscreteToBox(gym.ActionWrapper):
    """Convert discrete actions to continuous box space."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        n_actions = env.action_space.n
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(n_actions,), dtype=np.float32
        )
        self._original_space = env.action_space

    def action(self, action: NDArray[np.float32]) -> int:
        """Convert softmax-style continuous action to discrete."""
        return int(np.argmax(action))
```

### RewardWrapper

Transform reward signals:

```python
class RewardScale(gym.RewardWrapper):
    """Scale rewards by a constant factor."""

    def __init__(self, env: gym.Env, scale: float = 1.0) -> None:
        super().__init__(env)
        self._scale = scale

    def reward(self, reward: float) -> float:
        return reward * self._scale


class RewardClip(gym.RewardWrapper):
    """Clip rewards to a specified range."""

    def __init__(
        self, env: gym.Env, min_reward: float = -1.0, max_reward: float = 1.0
    ) -> None:
        super().__init__(env)
        self._min = min_reward
        self._max = max_reward

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, self._min, self._max))
```

### Using Wrappers with RLlib

Apply wrappers by wrapping the environment class:

```python
from ray.rllib.algorithms.ppo import PPOConfig


def env_creator(config: dict[str, Any]) -> gym.Env:
    """Create and wrap environment."""
    base_env = CustomEnv(config)
    wrapped = NormalizeObservation(base_env)
    wrapped = RewardScale(wrapped, scale=0.1)
    return wrapped


config = PPOConfig().environment(env_creator, env_config={"max_steps": 200})
```

## Observation Normalization

Normalize observations for stable training. Options include wrapper-based normalization (shown above) or built-in RLlib normalization.

### RLlib Built-in Normalization

Enable observation filtering in the algorithm config:

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(CustomEnv)
    .env_runners(
        observation_filter="MeanStdFilter",  # Running normalization
    )
)
```

### Manual Normalization in Environment

Implement normalization directly in the environment:

```python
class NormalizedEnv(gym.Env):
    """Environment with built-in observation normalization."""

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__()
        config = config or {}

        # Precomputed statistics (from data analysis)
        self._obs_mean = np.array([0.5, 1.2, -0.3, 2.1], dtype=np.float32)
        self._obs_std = np.array([0.8, 1.5, 0.4, 1.0], dtype=np.float32)

        # Normalized space
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)

    def _normalize(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize observation using precomputed statistics."""
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def reset(self, **kwargs: Any) -> tuple[NDArray[np.float32], dict[str, Any]]:
        raw_obs = self._get_raw_observation()
        return self._normalize(raw_obs), {}

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        # ... execute action ...
        raw_obs = self._get_raw_observation()
        return self._normalize(raw_obs), reward, terminated, truncated, {}
```

## Action Masking Techniques

Action masking restricts the agent to valid actions in each state, preventing illegal moves.

### Implementing Action Masking

Include action mask in the observation dict:

```python
class MaskedActionEnv(gym.Env):
    """Environment with dynamic action masking."""

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__()

        # Dict observation space with action_mask key
        self.observation_space = gym.spaces.Dict({
            "observations": gym.spaces.Box(-1.0, 1.0, (10,), np.float32),
            "action_mask": gym.spaces.Box(0, 1, (5,), np.int8),
        })
        self.action_space = gym.spaces.Discrete(5)

    def _get_action_mask(self) -> NDArray[np.int8]:
        """Return mask of valid actions (1=valid, 0=invalid)."""
        mask = np.ones(5, dtype=np.int8)
        # Mask invalid actions based on state
        if self._some_condition():
            mask[2] = 0  # Action 2 is invalid
            mask[4] = 0  # Action 4 is invalid
        return mask

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self._state = self._initialize_state()
        observation = {
            "observations": self._get_observation(),
            "action_mask": self._get_action_mask(),
        }
        return observation, {}

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        # Validate action against mask
        mask = self._get_action_mask()
        if mask[action] == 0:
            raise ValueError(f"Action {action} is masked (invalid)")

        # Execute action...
        observation = {
            "observations": self._get_observation(),
            "action_mask": self._get_action_mask(),
        }
        return observation, reward, terminated, truncated, {}
```

### Configuring RLlib for Action Masking

Specify the action mask key in AlgorithmConfig:

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(
        MaskedActionEnv,
        action_mask_key="action_mask",  # Key in observation dict
    )
)
```

RLlib automatically applies the mask during action sampling and policy updates.

## Environment Validation and Debugging

Comprehensive validation ensures environment correctness before training.

### Gymnasium check_env

Use the built-in checker for basic validation:

```python
from gymnasium.utils.env_checker import check_env

env = CustomEnv(config={"max_steps": 100})

# Validates spaces, reset, step, and rendering
check_env(env, skip_render_check=True)
```

### Custom Validation Suite

```python
import pytest
import numpy as np
from typing import Any


class TestEnvironmentCorrectness:
    """Comprehensive environment validation tests."""

    @pytest.fixture
    def env(self) -> CustomEnv:
        return CustomEnv(config={"max_steps": 50})

    def test_spaces_are_valid_gymnasium_spaces(self, env: CustomEnv) -> None:
        """Verify spaces are proper Gymnasium space instances."""
        import gymnasium.spaces as spaces

        assert isinstance(env.observation_space, spaces.Space)
        assert isinstance(env.action_space, spaces.Space)

    def test_reset_returns_valid_observation(self, env: CustomEnv) -> None:
        """Verify reset returns observation in space."""
        obs, info = env.reset()
        assert env.observation_space.contains(obs), (
            f"Reset observation {obs} not in {env.observation_space}"
        )
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self, env: CustomEnv) -> None:
        """Verify step returns exactly 5 values."""
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5, f"Expected 5-tuple, got {len(result)} values"

    def test_step_observation_in_space(self, env: CustomEnv) -> None:
        """Verify step observations are in observation space."""
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break

    def test_reward_is_scalar(self, env: CustomEnv) -> None:
        """Verify reward is a scalar float."""
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert isinstance(reward, (int, float, np.floating))
        assert np.isfinite(reward), f"Reward {reward} is not finite"

    def test_terminated_truncated_are_bool(self, env: CustomEnv) -> None:
        """Verify terminated and truncated are booleans."""
        env.reset()
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))

    def test_info_is_dict(self, env: CustomEnv) -> None:
        """Verify info is a dictionary."""
        _, info = env.reset()
        assert isinstance(info, dict)

        env.reset()
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert isinstance(info, dict)

    def test_episode_terminates_within_max_steps(self, env: CustomEnv) -> None:
        """Verify episode ends due to termination or truncation."""
        env.reset()
        max_iterations = 10000
        for i in range(max_iterations):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                return
        pytest.fail(f"Episode did not end within {max_iterations} steps")

    def test_deterministic_with_seed(self, env: CustomEnv) -> None:
        """Verify seeded reset produces reproducible results."""
        obs1, _ = env.reset(seed=12345)
        actions = [env.action_space.sample() for _ in range(5)]

        obs2, _ = env.reset(seed=12345)
        np.testing.assert_array_equal(obs1, obs2)

    def test_multiple_resets(self, env: CustomEnv) -> None:
        """Verify environment can be reset multiple times."""
        for _ in range(5):
            obs, info = env.reset()
            assert env.observation_space.contains(obs)
            # Take a few steps
            for _ in range(3):
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

    def test_action_space_sampling(self, env: CustomEnv) -> None:
        """Verify action space sampling works correctly."""
        for _ in range(100):
            action = env.action_space.sample()
            assert env.action_space.contains(action)
```

### Debugging Strategies

**Enable RLlib environment checking:**

```python
config = (
    PPOConfig()
    .environment(
        CustomEnv,
        disable_env_checking=False,  # Enable validation (default)
    )
)
```

**Add verbose logging to environment:**

```python
import logging

logger = logging.getLogger(__name__)


class DebugEnv(gym.Env):
    def step(self, action: int) -> tuple[...]:
        logger.debug(f"Step called with action={action}")
        result = self._internal_step(action)
        logger.debug(
            f"Step result: obs_shape={result[0].shape}, "
            f"reward={result[1]:.4f}, "
            f"terminated={result[2]}, truncated={result[3]}"
        )
        return result
```

**Validate with random policy before training:**

```python
def smoke_test_env(env_class: type, config: dict[str, Any], num_episodes: int = 5) -> None:
    """Run random episodes to catch obvious errors."""
    env = env_class(config)

    for ep in range(num_episodes):
        obs, info = env.reset()
        assert env.observation_space.contains(obs), f"Episode {ep}: Invalid reset obs"

        total_reward = 0.0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert env.observation_space.contains(obs), (
                f"Episode {ep}, step {steps}: Invalid step obs"
            )
            assert np.isfinite(reward), f"Non-finite reward: {reward}"

            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        print(f"Episode {ep}: steps={steps}, total_reward={total_reward:.2f}")

    env.close()
```

## RLlib Environment Configuration via AlgorithmConfig

Configure environment behavior through AlgorithmConfig methods.

### Basic Environment Setup

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(
        env=CustomEnv,  # Environment class or registered string
        env_config={  # Passed to env constructor
            "max_steps": 200,
            "difficulty": "hard",
        },
        observation_space=None,  # Auto-infer from env
        action_space=None,  # Auto-infer from env
        render_env=False,  # Disable rendering during training
        disable_env_checking=False,  # Enable validation
    )
)
```

### Reward and Action Processing

```python
config = (
    PPOConfig()
    .environment(
        env=CustomEnv,
        clip_rewards=True,  # Clip to [-1, 1]
        # Or specify range: clip_rewards=(-10.0, 10.0)
        normalize_actions=True,  # Learn in normalized action space
        clip_actions=True,  # Clip actions to space bounds
    )
)
```

### Action Masking Configuration

```python
config = (
    PPOConfig()
    .environment(
        env=MaskedActionEnv,
        action_mask_key="action_mask",  # Key in observation dict
    )
)
```

### Environment Runner Configuration

```python
config = (
    PPOConfig()
    .environment(CustomEnv, env_config={"max_steps": 200})
    .env_runners(
        num_env_runners=4,  # Parallel environment workers
        num_envs_per_env_runner=2,  # Envs per worker
        sample_timeout_s=60.0,  # Timeout for sampling
        observation_filter="MeanStdFilter",  # Observation normalization
        env_to_module_connector=None,  # Custom connector
        module_to_env_connector=None,  # Custom connector
    )
)
```

### Vectorization Configuration

```python
config = (
    PPOConfig()
    .environment(
        env="MyCustomEnv-v0",  # Registered environment
    )
    .env_runners(
        num_envs_per_env_runner=8,
    )
)
```

### Complete Example

```python
from ray.rllib.algorithms.ppo import PPOConfig
import ray


def main() -> None:
    ray.init()

    config = (
        PPOConfig()
        .environment(
            env=CustomEnv,
            env_config={
                "max_steps": 500,
                "reward_scale": 0.1,
            },
            clip_rewards=(-1.0, 1.0),
            normalize_actions=True,
        )
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=4,
            observation_filter="MeanStdFilter",
        )
        .training(
            train_batch_size_per_learner=4000,
            lr=3e-4,
            gamma=0.99,
        )
        .framework("torch")
    )

    algo = config.build()

    try:
        for i in range(100):
            result = algo.train()
            print(
                f"Iteration {i}: "
                f"reward={result['env_runners']['episode_reward_mean']:.2f}"
            )

            if i % 10 == 0:
                checkpoint_dir = algo.save()
                print(f"Checkpoint saved: {checkpoint_dir}")
    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
```

## Testing Fixtures for Environment Correctness

Reusable pytest fixtures for environment testing.

### Shared Fixtures Module

```python
# tests/conftest.py
from __future__ import annotations

from typing import Any, Generator

import pytest
import ray


@pytest.fixture(scope="module")
def ray_context() -> Generator[None, None, None]:
    """Initialize Ray for the test module."""
    ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def env_config() -> dict[str, Any]:
    """Default environment configuration."""
    return {"max_steps": 50}


@pytest.fixture
def env(env_config: dict[str, Any]) -> Generator[CustomEnv, None, None]:
    """Create environment instance."""
    environment = CustomEnv(config=env_config)
    yield environment
    environment.close()


@pytest.fixture
def seeded_env(env_config: dict[str, Any]) -> Generator[CustomEnv, None, None]:
    """Create seeded environment for reproducible tests."""
    environment = CustomEnv(config=env_config)
    environment.reset(seed=42)
    yield environment
    environment.close()
```

### Parameterized Space Tests

```python
import pytest
import gymnasium as gym
import numpy as np


@pytest.mark.parametrize("space,expected_shape", [
    (gym.spaces.Box(-1, 1, (4,), np.float32), (4,)),
    (gym.spaces.Box(-1, 1, (2, 3), np.float32), (2, 3)),
    (gym.spaces.Discrete(5), ()),
])
def test_space_shapes(space: gym.spaces.Space, expected_shape: tuple[int, ...]) -> None:
    """Verify space shape expectations."""
    sample = space.sample()
    assert np.array(sample).shape == expected_shape


@pytest.mark.parametrize("config", [
    {"max_steps": 10},
    {"max_steps": 100},
    {"max_steps": 1000},
])
def test_env_with_various_configs(config: dict[str, Any]) -> None:
    """Test environment with different configurations."""
    env = CustomEnv(config=config)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    env.close()
```

### Smoke Training Test

```python
import pytest
from ray.rllib.algorithms.ppo import PPOConfig


@pytest.mark.slow
def test_smoke_training(ray_context: None) -> None:
    """Verify environment works with RLlib training."""
    config = (
        PPOConfig()
        .environment(CustomEnv, env_config={"max_steps": 50})
        .env_runners(num_env_runners=0)  # Local only for test
        .training(
            train_batch_size_per_learner=200,
            num_epochs=1,
            minibatch_size=50,
        )
        .framework("torch")
    )

    algo = config.build()
    try:
        # Run 2 training iterations
        for _ in range(2):
            result = algo.train()
            assert "env_runners" in result
            assert "episode_reward_mean" in result["env_runners"]
    finally:
        algo.stop()
```
