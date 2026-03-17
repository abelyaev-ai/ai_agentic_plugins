---
name: rllib-training-pipeline
description: >
  This skill should be used when the user asks to "train an RLlib algorithm",
  "set up AlgorithmConfig", "checkpoint a training run", "evaluate an RL policy",
  or builds RLlib training loops.
---

## Purpose

Provide patterns and best practices for the complete RLlib training lifecycle using the new API stack (Ray 2.53+). This skill covers algorithm configuration, training loop implementation, checkpoint management, evaluation workflows, and resource specification for both local and distributed training scenarios.

## Prerequisites

Before implementing training pipelines, resolve the RLlib documentation context:

```
context7: resolve-library-id "ray rllib"
context7: query-docs /websites/ray_io_en_master "AlgorithmConfig training checkpoint evaluation"
```

Verify Ray and RLlib versions are compatible with the new API stack:

```python
import ray
print(f"Ray version: {ray.__version__}")  # Require >= 2.53
```

## Core Workflow

### AlgorithmConfig Builder Pattern

The foundation of RLlib training is the `AlgorithmConfig` builder pattern. Never use dictionary-based configuration or legacy `Trainer` classes. Always construct configuration using the fluent builder API.

#### Environment Configuration

Start by specifying the environment. Accept registered environment names, environment classes, or factory functions:

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = PPOConfig()
config.environment(
    env="CartPole-v1",                    # Registered env name
    env_config={"max_episode_steps": 500}, # Env-specific config
    observation_space=None,                # Override if needed
    action_space=None,                     # Override if needed
)
```

For custom environments, pass the class directly:

```python
from my_envs import CustomTradingEnv

config.environment(
    env=CustomTradingEnv,
    env_config={"symbol": "AAPL", "window_size": 20},
)
```

#### Training Hyperparameters

Configure training parameters using the `.training()` method. Key parameters vary by algorithm but common ones include:

```python
config.training(
    gamma=0.99,                          # Discount factor
    lr=3e-4,                             # Learning rate (or schedule)
    train_batch_size_per_learner=4000,   # Batch size per learner
    num_epochs=10,                       # SGD epochs per iteration
    minibatch_size=128,                  # Minibatch size for SGD
    grad_clip=0.5,                       # Gradient clipping
    grad_clip_by="global_norm",          # Clipping method
)
```

For learning rate schedules, provide a list of `[timestep, lr_value]` pairs:

```python
config.training(
    lr=[[0, 1e-3], [100000, 5e-4], [500000, 1e-4]],  # Linear interpolation
)
```

#### Environment Runners Configuration

Configure parallel environment sampling with `.env_runners()`. Use `num_env_runners` (never the deprecated `num_workers`):

```python
config.env_runners(
    num_env_runners=4,                   # Parallel sampling workers
    num_envs_per_env_runner=1,           # Envs per worker
    sample_timeout_s=60.0,               # Sampling timeout
    rollout_fragment_length="auto",      # Fragment length
)
```

#### Learner Configuration

Configure learner resources for distributed gradient computation:

```python
config.learners(
    num_learners=0,                      # 0 = local, >0 = distributed
    num_gpus_per_learner=0,              # GPUs per learner (0 or 1)
    num_cpus_per_learner=1,              # CPUs per learner
)
```

For GPU training with DDP-style updates:

```python
config.learners(
    num_learners=2,                      # 2 distributed learners
    num_gpus_per_learner=1,              # 1 GPU each
)
```

#### Framework Specification

Always specify PyTorch as the framework. TensorFlow is not supported on the new API stack:

```python
config.framework("torch")
```

### Building the Algorithm Instance

After configuration, call `.build()` to create the algorithm instance:

```python
algo = config.build()
```

The `build()` method validates configuration, initializes Ray actors for env runners and learners, creates the RLModule (neural network), and sets up replay buffers if applicable.

Handle build failures with proper error handling:

```python
import ray
from ray.rllib.algorithms.ppo import PPOConfig

ray.init()

try:
    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .env_runners(num_env_runners=2)
        .training(train_batch_size_per_learner=2000)
        .framework("torch")
    )
    algo = config.build()
except Exception as e:
    ray.shutdown()
    raise RuntimeError(f"Failed to build algorithm: {e}") from e
```

### Training Loop Implementation

Execute training iterations with `algo.train()`. Each call performs one complete training iteration including sampling, learning, and metric computation:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

num_iterations = 100
best_reward = float("-inf")

for i in range(num_iterations):
    result = algo.train()

    # Extract key metrics
    episode_reward_mean = result.get("env_runners", {}).get("episode_return_mean", 0)
    episode_len_mean = result.get("env_runners", {}).get("episode_len_mean", 0)

    logger.info(
        f"Iteration {i}: reward={episode_reward_mean:.2f}, "
        f"length={episode_len_mean:.2f}"
    )

    # Track best performance
    if episode_reward_mean > best_reward:
        best_reward = episode_reward_mean
```

### Checkpoint Management

Save checkpoints periodically and at key milestones. Use `algo.save_checkpoint()` for manual saves:

```python
import os
from pathlib import Path

checkpoint_dir = Path("./checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

for i in range(num_iterations):
    result = algo.train()
    episode_reward_mean = result.get("env_runners", {}).get("episode_return_mean", 0)

    # Periodic checkpoint
    if (i + 1) % 10 == 0:
        checkpoint_path = algo.save_checkpoint(str(checkpoint_dir))
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Best model checkpoint
    if episode_reward_mean > best_reward:
        best_reward = episode_reward_mean
        best_checkpoint_path = algo.save_checkpoint(str(checkpoint_dir / "best"))
        logger.info(f"New best model: reward={best_reward:.2f}")
```

### Checkpoint Restoration

Restore from checkpoint using `Algorithm.from_checkpoint()`:

```python
from ray.rllib.algorithms.ppo import PPO

# Restore algorithm from checkpoint
restored_algo = PPO.from_checkpoint(checkpoint_path)

# Continue training
for i in range(additional_iterations):
    result = restored_algo.train()
```

Alternatively, restore weights into an existing algorithm:

```python
algo.restore_from_path(checkpoint_path)
```

### Evaluation

Run evaluation with `algo.evaluate()` to assess policy performance without training:

```python
# Configure evaluation during setup
config.evaluation(
    evaluation_interval=5,               # Evaluate every 5 iterations
    evaluation_num_env_runners=2,        # Dedicated eval workers
    evaluation_duration=10,              # Episodes per evaluation
    evaluation_config={
        "explore": False,                # Disable exploration
    },
)

# Manual evaluation
eval_results = algo.evaluate()
eval_reward = eval_results.get("env_runners", {}).get("episode_return_mean", 0)
logger.info(f"Evaluation reward: {eval_reward:.2f}")
```

### Stopping Criteria

Implement stopping criteria based on performance thresholds or iteration limits:

```python
target_reward = 195.0  # CartPole solved threshold
max_iterations = 500
patience = 20
no_improvement_count = 0

for i in range(max_iterations):
    result = algo.train()
    episode_reward_mean = result.get("env_runners", {}).get("episode_return_mean", 0)

    # Check target reached
    if episode_reward_mean >= target_reward:
        logger.info(f"Target reward reached at iteration {i}")
        break

    # Early stopping with patience
    if episode_reward_mean > best_reward:
        best_reward = episode_reward_mean
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if no_improvement_count >= patience:
        logger.info(f"Early stopping: no improvement for {patience} iterations")
        break
```

### Resource Cleanup

Always clean up resources properly when training completes:

```python
try:
    # Training loop
    for i in range(num_iterations):
        result = algo.train()
        # ... process results
finally:
    algo.stop()
    ray.shutdown()
```

### Complete Training Template

Combine all components into a production-ready training script:

```python
import logging
import ray
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig, PPO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_ppo(
    env_name: str = "CartPole-v1",
    num_iterations: int = 100,
    checkpoint_dir: str = "./checkpoints",
    target_reward: float = 195.0,
) -> str:
    """Train PPO algorithm with checkpointing and early stopping."""
    ray.init()
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    best_reward = float("-inf")
    best_checkpoint = None

    try:
        config = (
            PPOConfig()
            .environment(env_name)
            .env_runners(num_env_runners=2)
            .learners(num_learners=0, num_gpus_per_learner=0)
            .training(
                gamma=0.99,
                lr=3e-4,
                train_batch_size_per_learner=4000,
                num_epochs=10,
                minibatch_size=128,
            )
            .framework("torch")
        )

        algo = config.build()

        for i in range(num_iterations):
            result = algo.train()
            episode_reward_mean = result.get("env_runners", {}).get(
                "episode_return_mean", 0
            )

            logger.info(f"Iteration {i}: reward={episode_reward_mean:.2f}")

            # Save best checkpoint
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                best_checkpoint = algo.save_checkpoint(
                    str(checkpoint_path / "best")
                )

            # Check stopping criterion
            if episode_reward_mean >= target_reward:
                logger.info(f"Target reached at iteration {i}")
                break

        return best_checkpoint

    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    checkpoint = train_ppo()
    print(f"Best checkpoint: {checkpoint}")
```

## Common Pitfalls

- **Using deprecated API**: Never use `PPOTrainer`, `num_workers`, `train_batch_size` (old style), or TensorFlow framework. Always use `PPOConfig().build()`, `num_env_runners`, `train_batch_size_per_learner`, and `framework("torch")`.

- **Missing resource cleanup**: Always wrap training in try/finally with `algo.stop()` and `ray.shutdown()` in the finally block. Leaked Ray processes cause memory issues and port conflicts.

- **Ignoring checkpoint frequency**: Checkpointing too frequently wastes disk space and slows training. Checkpointing too rarely risks losing progress on crashes. Balance with periodic saves (every 10-50 iterations) plus best-model saves.

- **Incorrect metric paths**: New API stack metrics are nested under `env_runners` key (e.g., `result["env_runners"]["episode_return_mean"]`). Old tutorials may show flat metric names that no longer exist.

- **Misconfigured distributed resources**: When using `num_learners > 0`, ensure `num_gpus_per_learner` matches available GPUs. Requesting unavailable resources causes silent hangs or cryptic errors.

## Additional Resources

For detailed algorithm-specific configurations, callback patterns, curriculum learning, and Ray Tune integration, see `references/training-workflows.md`.
