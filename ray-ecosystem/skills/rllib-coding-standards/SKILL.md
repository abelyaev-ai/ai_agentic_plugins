---
name: rllib-coding-standards
description: >
  This skill should be used when the user asks about "RLlib new API",
  "RLlib anti-patterns", "migrate from old RLlib API", "RLlib type safety",
  or needs RLlib coding conventions (2.53+).
---

## Purpose

Provide coding standards, anti-patterns, and migration guidance for RLlib new API stack (2.53+). This skill covers the modern AlgorithmConfig builder pattern, RLModule architecture, EnvRunner terminology, type safety requirements, error handling conventions, and testing best practices. Use this skill to ensure RLlib code follows current best practices and avoids deprecated patterns.

## Prerequisites

Before implementing RLlib solutions, resolve the latest RLlib documentation via Context7:

1. Call `resolve-library-id` with `libraryName: "ray rllib"` and query describing the migration or API topic
2. Call `query-docs` with the resolved library ID (typically `/websites/ray_io_en_master`) and specific API questions
3. Ground all recommendations in the latest API surface rather than memorized patterns

## Core Workflow

### New API Requirements (Ray 2.53+)

The new API stack is the default and mandatory path for all new RLlib development. Follow these requirements strictly:

#### AlgorithmConfig Builders

Use the fluent builder pattern with algorithm-specific Config classes. Never use dictionary-based configuration.

```python
# CORRECT: AlgorithmConfig builder pattern
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .framework("torch")
    .env_runners(num_env_runners=4)
    .training(
        gamma=0.99,
        lr=0.0003,
        train_batch_size_per_learner=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
    )
    .resources(num_gpus=0)
)

algo = config.build()
```

#### EnvRunner Terminology

Replace all `worker` terminology with `EnvRunner`:

| Old Term | New Term |
|----------|----------|
| `num_workers` | `num_env_runners` |
| `num_rollout_workers` | `num_env_runners` |
| `RolloutWorker` | `EnvRunner` |
| `num_cpus_per_worker` | `num_cpus_per_env_runner` |
| `num_gpus_per_worker` | `num_gpus_per_env_runner` |

```python
# CORRECT: EnvRunner configuration
config.env_runners(
    num_env_runners=4,
    num_cpus_per_env_runner=1,
    num_gpus_per_env_runner=0,
)
```

#### Learner-Centric Batching

Use `train_batch_size_per_learner` instead of the old `train_batch_size`. This parameter specifies the batch size processed by each Learner actor, enabling better scaling semantics.

```python
# CORRECT: Learner-centric batch configuration
config.training(
    train_batch_size_per_learner=4000,  # Per-learner batch size
    sgd_minibatch_size=128,
    num_sgd_iter=10,
)
```

#### PyTorch Only

The new API stack exclusively supports PyTorch. Do not use TensorFlow or TensorFlow 2.

```python
# CORRECT: PyTorch framework
config.framework("torch")

# FORBIDDEN: TensorFlow frameworks
# config.framework("tf")      # NEVER use
# config.framework("tf2")     # NEVER use
```

#### RLModule Architecture

Replace `ModelV2` and `TorchModelV2` with `RLModule` and `TorchRLModule`. The RLModule API provides cleaner separation between inference and training forward passes.

```python
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns
import torch

class CustomTorchRLModule(TorchRLModule):
    def setup(self):
        # Access pre-set attributes:
        # self.observation_space, self.action_space
        # self.inference_only, self.model_config
        input_dim = self.observation_space.shape[0]
        hidden_dim = self.model_config.get("hidden_dim", 64)
        output_dim = self.action_space.n

        self._policy_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def _forward(self, batch, **kwargs):
        action_logits = self._policy_net(batch[Columns.OBS])
        return {Columns.ACTION_DIST_INPUTS: action_logits}

    def _forward_inference(self, batch, **kwargs):
        # Deterministic action selection for deployment
        return self._forward(batch, **kwargs)

    def _forward_exploration(self, batch, **kwargs):
        # Stochastic action selection for training
        return self._forward(batch, **kwargs)
```

### Type Safety Requirements

Apply strict type hints to all public functions and methods.

#### Function Signatures

```python
from typing import Optional, Any
import numpy as np
from numpy.typing import NDArray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.env_context import EnvContext
import gymnasium as gym

def create_environment(
    env_config: dict[str, Any],
    env_context: Optional[EnvContext] = None,
) -> gym.Env:
    """Create and return a configured environment instance."""
    ...

def process_observations(
    obs: NDArray[np.float32],
    normalize: bool = True,
) -> NDArray[np.float32]:
    """Process raw observations with optional normalization."""
    ...

def train_algorithm(
    algo: Algorithm,
    num_iterations: int,
    checkpoint_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Train algorithm for specified iterations with optional checkpointing."""
    ...
```

#### Type Conventions

- Use `Optional[T]` where `None` is a valid value (required for RLlib stub compatibility)
- Use `numpy.typing.NDArray[np.float32]` for array types in public APIs
- Use `dict[str, Any]` (modern Python style) over `Dict[str, Any]`
- Accept `Dict` for RLlib compatibility when interfacing with RLlib internals
- Always annotate return types, including `-> None` for procedures

### Error Handling (EAFP Style)

Follow Easier to Ask for Forgiveness than Permission (EAFP) conventions with specific, tight exception handling.

#### Specific Exceptions

Never use bare `except:` clauses. Always catch specific exception types.

```python
# CORRECT: Specific exception handling
from ray.rllib.utils.error import EnvError

try:
    obs, info = env.reset()
except EnvError as e:
    raise EnvironmentInitError(f"Failed to reset environment: {env}") from e
except ValueError as e:
    raise ConfigurationError(f"Invalid environment config") from e
```

#### Exception Chaining

Always chain exceptions to preserve the original traceback.

```python
# CORRECT: Exception chaining
try:
    result = algo.train()
except RuntimeError as e:
    raise TrainingError(f"Training iteration failed") from e
```

#### Tight Try/Except Blocks

Keep try blocks minimal (2-5 lines) to catch only the expected exceptions.

```python
# CORRECT: Tight try/except
def load_checkpoint(path: str) -> Algorithm:
    try:
        algo = Algorithm.from_checkpoint(path)
    except FileNotFoundError as e:
        raise CheckpointError(f"Checkpoint not found: {path}") from e

    # Validation outside try block
    if algo is None:
        raise CheckpointError("Loaded algorithm is None")
    return algo
```

#### Cleanup in Finally

Use `finally` blocks for resource cleanup, especially for Ray resources.

```python
# CORRECT: Cleanup in finally
import ray
from ray.rllib.algorithms.ppo import PPOConfig

algo = None
try:
    ray.init()
    config = PPOConfig().environment("CartPole-v1")
    algo = config.build()
    result = algo.train()
finally:
    if algo is not None:
        algo.stop()
    ray.shutdown()
```

### Testing Best Practices

#### Module-Scoped Fixtures

Use module-scoped fixtures for Ray initialization to avoid repeated startup overhead.

```python
import pytest
import ray

@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for the test module."""
    ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()

def test_training_smoke(ray_context):
    """Smoke test: verify training runs without errors."""
    from ray.rllib.algorithms.ppo import PPOConfig

    config = PPOConfig().environment("CartPole-v1").env_runners(num_env_runners=0)
    algo = config.build()
    try:
        result = algo.train()
        assert "env_runners" in result
    finally:
        algo.stop()
```

#### Environment Tests

Validate environment compliance with Gymnasium 5-tuple protocol.

```python
import pytest
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

def test_env_spaces():
    """Verify observation and action space types."""
    env = gym.make("CartPole-v1")
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)
    env.close()

def test_env_reset_returns_tuple():
    """Verify reset returns (obs, info) tuple."""
    env = gym.make("CartPole-v1")
    result = env.reset()
    assert isinstance(result, tuple)
    assert len(result) == 2
    obs, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    env.close()

def test_env_step_returns_5tuple():
    """Verify step returns (obs, reward, terminated, truncated, info)."""
    env = gym.make("CartPole-v1")
    env.reset()
    action = env.action_space.sample()
    result = env.step(action)
    assert isinstance(result, tuple)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()
```

#### Training Tests

Include smoke training tests with checkpoint save/load verification.

```python
import pytest
import tempfile
from pathlib import Path

def test_checkpoint_save_load(ray_context):
    """Verify checkpoint save and restore functionality."""
    from ray.rllib.algorithms.ppo import PPOConfig

    config = PPOConfig().environment("CartPole-v1").env_runners(num_env_runners=0)
    algo = config.build()

    try:
        # Train briefly
        algo.train()

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = algo.save(tmpdir)
            assert Path(checkpoint_path).exists()

            # Restore and verify
            restored = config.build()
            restored.restore(checkpoint_path)
            restored.stop()
    finally:
        algo.stop()

@pytest.mark.parametrize("env_name", ["CartPole-v1", "Pendulum-v1"])
def test_training_multiple_envs(ray_context, env_name):
    """Smoke test training across multiple environments."""
    from ray.rllib.algorithms.ppo import PPOConfig

    config = PPOConfig().environment(env_name).env_runners(num_env_runners=0)
    algo = config.build()
    try:
        result = algo.train()
        assert result is not None
    finally:
        algo.stop()
```

### Anti-Patterns (Forbidden)

#### Old API Classes

| Forbidden | Replacement |
|-----------|-------------|
| `PPOTrainer`, `DQNTrainer`, `SACTrainer` | `PPOConfig().build()`, `DQNConfig().build()`, `SACConfig().build()` |
| `ModelV2`, `TorchModelV2`, `TFModelV2` | `TorchRLModule` |
| `RolloutWorker` | `EnvRunner` |
| `Policy` (for custom models) | `RLModule` |

#### Old Configuration Parameters

| Forbidden | Replacement |
|-----------|-------------|
| `num_workers` | `num_env_runners` |
| `num_rollout_workers` | `num_env_runners` |
| `train_batch_size` (old semantics) | `train_batch_size_per_learner` |
| `framework="tf"` | `framework="torch"` |
| `framework="tf2"` | `framework="torch"` |

#### General Python Anti-Patterns

```python
# FORBIDDEN: Bare except
try:
    algo.train()
except:  # NEVER do this
    pass

# FORBIDDEN: Missing type hints
def process(data):  # Missing type annotations
    return data

# FORBIDDEN: print() for logging
print(f"Training result: {result}")  # Use logging module instead

# FORBIDDEN: Mutable default arguments
def train(config={}):  # Mutable default
    ...

# CORRECT alternatives:
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def process(data: NDArray[np.float32]) -> NDArray[np.float32]:
    return data

def train(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if config is None:
        config = {}
    ...
    logger.info(f"Training result: {result}")
```

## Common Pitfalls

- **Mixing API stacks**: Do not combine new API components (RLModule) with old API components (ModelV2) in the same algorithm. The stacks are incompatible.

- **Forgetting `algo.stop()`**: Always call `algo.stop()` in a `finally` block to release Ray resources. Leaked workers cause memory issues and port conflicts.

- **Using old batch size semantics**: The old `train_batch_size` specified the total batch across all workers. The new `train_batch_size_per_learner` specifies per-learner batch size. Adjust values accordingly when migrating.

- **Ignoring Gymnasium 5-tuple**: Custom environments must return `(obs, reward, terminated, truncated, info)` from `step()`. The old 4-tuple `(obs, reward, done, info)` causes runtime errors.

- **Hardcoding resource counts**: Use `num_env_runners=0` for local debugging and testing. Scale via configuration for production rather than hardcoded values.

## Additional Resources

For detailed migration examples and version compatibility information, consult:

- **API Migration Reference**: `references/api-migration.md` in this skill directory
- **RLlib Documentation**: https://docs.ray.io/en/latest/rllib/
- **Migration Guide**: https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html
- **RLModule API**: https://docs.ray.io/en/latest/rllib/rl-modules.html
