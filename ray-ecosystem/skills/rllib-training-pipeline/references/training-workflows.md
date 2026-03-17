# RLlib Training Workflows Reference

This reference provides detailed patterns for algorithm-specific configuration, advanced training techniques, and production deployment workflows using the RLlib new API stack (2.53+).

## Algorithm-Specific Configuration Examples

### Proximal Policy Optimization (PPO)

PPO is the recommended starting algorithm for most continuous and discrete control tasks. It balances sample efficiency with stability through clipped surrogate objectives.

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("HalfCheetah-v4")
    .env_runners(
        num_env_runners=8,               # Parallel sampling
        num_envs_per_env_runner=1,
        rollout_fragment_length=200,     # Steps per fragment
    )
    .learners(
        num_learners=1,                  # Single GPU learner
        num_gpus_per_learner=1,
    )
    .training(
        # Core PPO hyperparameters
        gamma=0.99,                      # Discount factor
        lr=3e-4,                         # Learning rate
        lambda_=0.95,                    # GAE lambda
        kl_coeff=0.2,                    # KL penalty coefficient
        clip_param=0.2,                  # PPO clip parameter

        # Batch configuration
        train_batch_size_per_learner=4000,
        num_epochs=10,                   # SGD passes per batch
        minibatch_size=128,              # Minibatch for SGD

        # Gradient handling
        grad_clip=0.5,
        grad_clip_by="global_norm",

        # Entropy bonus for exploration
        entropy_coeff=0.01,
        entropy_coeff_schedule=[[0, 0.01], [1000000, 0.001]],

        # Value function configuration
        vf_loss_coeff=0.5,
        vf_clip_param=10.0,
    )
    .framework("torch")
)

algo = config.build()
```

**Key PPO Hyperparameters:**

| Parameter | Typical Range | Purpose |
|-----------|--------------|---------|
| `clip_param` | 0.1 - 0.3 | Controls policy update magnitude |
| `kl_coeff` | 0.0 - 1.0 | Adaptive KL penalty weight |
| `num_epochs` | 3 - 30 | SGD iterations per batch |
| `lambda_` | 0.9 - 1.0 | GAE bias-variance tradeoff |
| `entropy_coeff` | 0.0 - 0.1 | Exploration encouragement |

### Soft Actor-Critic (SAC)

SAC excels in continuous control with automatic entropy tuning. It is off-policy, enabling better sample efficiency through replay buffer usage.

```python
from ray.rllib.algorithms.sac import SACConfig

config = (
    SACConfig()
    .environment("Pendulum-v1")
    .env_runners(
        num_env_runners=4,
        rollout_fragment_length=1,       # Step-based for off-policy
    )
    .learners(
        num_learners=0,                  # Local learner
        num_gpus_per_learner=1,
    )
    .training(
        # Discount and target network
        gamma=0.99,
        tau=0.005,                       # Soft target update rate

        # Learning rates (separate for actor/critic)
        actor_lr=3e-4,
        critic_lr=3e-4,
        entropy_lr=3e-4,                 # Alpha learning rate

        # Batch configuration
        train_batch_size_per_learner=256,
        num_epochs=1,                    # Single pass typical for SAC

        # Replay buffer
        replay_buffer_config={
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": 1000000,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
        },

        # Learning starts
        num_steps_sampled_before_learning_starts=10000,

        # Target entropy (auto-tuned if None)
        target_entropy="auto",

        # N-step returns
        n_step=1,
    )
    .framework("torch")
)

algo = config.build()
```

**Key SAC Hyperparameters:**

| Parameter | Typical Range | Purpose |
|-----------|--------------|---------|
| `tau` | 0.001 - 0.01 | Soft target update rate |
| `actor_lr` | 1e-4 - 1e-3 | Actor network learning rate |
| `critic_lr` | 1e-4 - 1e-3 | Critic network learning rate |
| `target_entropy` | "auto" or float | Entropy target for alpha tuning |
| `n_step` | 1 - 5 | Multi-step return horizon |

### Deep Q-Network (DQN)

DQN is the foundation for discrete action space problems. Use for environments with finite action sets.

```python
from ray.rllib.algorithms.dqn import DQNConfig

config = (
    DQNConfig()
    .environment("CartPole-v1")
    .env_runners(
        num_env_runners=2,
        rollout_fragment_length=4,
    )
    .learners(
        num_learners=0,
        num_gpus_per_learner=0,
    )
    .training(
        # Core DQN parameters
        gamma=0.99,
        lr=5e-4,

        # Batch configuration
        train_batch_size_per_learner=32,

        # Target network
        target_network_update_freq=500,  # Steps between target updates

        # Replay buffer
        replay_buffer_config={
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": 50000,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
        },

        # Learning starts
        num_steps_sampled_before_learning_starts=1000,

        # Double Q-learning
        double_q=True,

        # Dueling architecture
        dueling=True,

        # N-step returns
        n_step=3,

        # Noisy networks (alternative to epsilon-greedy)
        noisy=False,
    )
    .exploration(
        exploration_config={
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 10000,
        }
    )
    .framework("torch")
)

algo = config.build()
```

**Key DQN Hyperparameters:**

| Parameter | Typical Range | Purpose |
|-----------|--------------|---------|
| `target_network_update_freq` | 100 - 10000 | Target network sync frequency |
| `double_q` | True/False | Double DQN for overestimation |
| `dueling` | True/False | Dueling architecture |
| `n_step` | 1 - 5 | Multi-step return horizon |
| `noisy` | True/False | Noisy networks for exploration |

## Curriculum Learning Setup

Curriculum learning progressively increases task difficulty as the agent improves. Implement using custom callbacks that modify the environment based on training progress.

### Callback-Based Curriculum

```python
from ray.rllib.callbacks.callbacks import RLlibCallback
from typing import Any


class CurriculumCallback(RLlibCallback):
    """Adjust environment difficulty based on training progress."""

    def __init__(self):
        super().__init__()
        self.current_level = 0
        self.level_thresholds = [50, 100, 150, 180]  # Reward thresholds

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict[str, Any],
        **kwargs,
    ) -> None:
        """Update curriculum level after each training iteration."""
        episode_reward_mean = result.get("env_runners", {}).get(
            "episode_return_mean", 0
        )

        # Determine new level based on performance
        new_level = 0
        for i, threshold in enumerate(self.level_thresholds):
            if episode_reward_mean >= threshold:
                new_level = i + 1

        # Update environment if level changed
        if new_level != self.current_level:
            self.current_level = new_level
            algorithm.env_runner_group.foreach_worker(
                lambda worker: worker.foreach_env(
                    lambda env: env.set_difficulty(new_level)
                )
            )
            print(f"Curriculum advanced to level {new_level}")


# Use in configuration
config = (
    PPOConfig()
    .environment(CurriculumEnv)
    .callbacks(CurriculumCallback)
)
```

### Environment with Curriculum Support

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray


class CurriculumEnv(gym.Env):
    """Environment supporting curriculum learning."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.difficulty = 0
        self.difficulty_params = {
            0: {"obstacle_speed": 0.5, "num_obstacles": 2},
            1: {"obstacle_speed": 1.0, "num_obstacles": 3},
            2: {"obstacle_speed": 1.5, "num_obstacles": 4},
            3: {"obstacle_speed": 2.0, "num_obstacles": 5},
            4: {"obstacle_speed": 2.5, "num_obstacles": 6},
        }
        self._apply_difficulty()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

    def set_difficulty(self, level: int) -> None:
        """Set curriculum difficulty level."""
        self.difficulty = min(level, len(self.difficulty_params) - 1)
        self._apply_difficulty()

    def _apply_difficulty(self) -> None:
        """Apply current difficulty parameters."""
        params = self.difficulty_params[self.difficulty]
        self.obstacle_speed = params["obstacle_speed"]
        self.num_obstacles = params["num_obstacles"]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[NDArray[np.float32], dict]:
        super().reset(seed=seed)
        # Reset logic using current difficulty
        obs = np.zeros(10, dtype=np.float32)
        return obs, {"difficulty": self.difficulty}

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        # Step logic using self.obstacle_speed, self.num_obstacles
        obs = np.zeros(10, dtype=np.float32)
        reward = 1.0
        terminated = False
        truncated = False
        info = {"difficulty": self.difficulty}
        return obs, reward, terminated, truncated, info
```

## Custom Callbacks

Callbacks hook into the training lifecycle for logging, metrics, early stopping, and custom logic.

### Comprehensive Callback Example

```python
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from typing import Any
import logging

logger = logging.getLogger(__name__)


class TrainingMonitorCallback(RLlibCallback):
    """Comprehensive callback for training monitoring."""

    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self.best_reward = float("-inf")

    def on_episode_start(
        self,
        *,
        episode: EpisodeV2,
        env_runner: "EnvRunner",
        metrics_logger: "MetricsLogger",
        env: BaseEnv,
        env_index: int,
        rl_module: "RLModule",
        **kwargs,
    ) -> None:
        """Called at the start of each episode."""
        episode.user_data["custom_metric"] = 0
        episode.hist_data["action_distribution"] = []

    def on_episode_step(
        self,
        *,
        episode: EpisodeV2,
        env_runner: "EnvRunner",
        metrics_logger: "MetricsLogger",
        env: BaseEnv,
        env_index: int,
        rl_module: "RLModule",
        **kwargs,
    ) -> None:
        """Called at each environment step."""
        # Track custom metrics during episode
        action = episode.get_actions(-1)
        if action is not None:
            episode.hist_data["action_distribution"].append(int(action))

        # Accumulate custom metric
        episode.user_data["custom_metric"] += 1

    def on_episode_end(
        self,
        *,
        episode: EpisodeV2,
        env_runner: "EnvRunner",
        metrics_logger: "MetricsLogger",
        env: BaseEnv,
        env_index: int,
        rl_module: "RLModule",
        **kwargs,
    ) -> None:
        """Called at the end of each episode."""
        total_reward = episode.total_reward
        episode_length = episode.length

        # Log custom metrics
        episode.custom_metrics["episode_total_steps"] = episode_length
        episode.custom_metrics["custom_accumulated"] = episode.user_data[
            "custom_metric"
        ]

        # Compute action entropy
        actions = episode.hist_data.get("action_distribution", [])
        if actions:
            unique, counts = np.unique(actions, return_counts=True)
            probs = counts / len(actions)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            episode.custom_metrics["action_entropy"] = entropy

        self.episode_rewards.append(total_reward)

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict[str, Any],
        **kwargs,
    ) -> None:
        """Called after each training iteration."""
        episode_reward_mean = result.get("env_runners", {}).get(
            "episode_return_mean", 0
        )

        # Track best performance
        if episode_reward_mean > self.best_reward:
            self.best_reward = episode_reward_mean
            result["custom_metrics"] = result.get("custom_metrics", {})
            result["custom_metrics"]["best_reward"] = self.best_reward

        # Log training progress
        iteration = result.get("training_iteration", 0)
        logger.info(
            f"Iteration {iteration}: "
            f"reward={episode_reward_mean:.2f}, "
            f"best={self.best_reward:.2f}"
        )

        # Early stopping check (implement custom logic)
        if episode_reward_mean >= 195.0:
            result["done"] = True  # Signal to stop training


import numpy as np
```

### Registering Callbacks

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .callbacks(TrainingMonitorCallback)
)
```

## Training Metrics Interpretation

Understanding training metrics is essential for diagnosing issues and tuning hyperparameters.

### Key Metric Categories

**Environment Runner Metrics** (under `result["env_runners"]`):

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `episode_return_mean` | Average episode reward | Task-dependent |
| `episode_return_max` | Maximum episode reward | Should increase |
| `episode_return_min` | Minimum episode reward | Should stabilize |
| `episode_len_mean` | Average episode length | Task-dependent |
| `num_env_steps_sampled` | Steps in this iteration | Matches config |
| `num_episodes_sampled` | Episodes in this iteration | Varies |

**Learner Metrics** (under `result["learner"]`):

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `total_loss` | Combined loss value | Should decrease |
| `policy_loss` | Policy gradient loss | Should decrease |
| `vf_loss` | Value function loss | Should decrease |
| `entropy` | Policy entropy | Gradual decrease |
| `kl_divergence` | KL from old policy | < kl_target |
| `grad_norm` | Gradient magnitude | Stable, not exploding |

### Metric Extraction Pattern

```python
def extract_metrics(result: dict) -> dict:
    """Extract key metrics from training result."""
    env_metrics = result.get("env_runners", {})
    learner_metrics = result.get("learner", {})

    return {
        # Performance
        "reward_mean": env_metrics.get("episode_return_mean", 0),
        "reward_max": env_metrics.get("episode_return_max", 0),
        "episode_len": env_metrics.get("episode_len_mean", 0),

        # Learning
        "total_loss": learner_metrics.get("total_loss", 0),
        "policy_loss": learner_metrics.get("policy_loss", 0),
        "vf_loss": learner_metrics.get("vf_loss", 0),
        "entropy": learner_metrics.get("entropy", 0),

        # Throughput
        "steps_sampled": env_metrics.get("num_env_steps_sampled", 0),
        "timesteps_total": result.get("timesteps_total", 0),
    }
```

## Checkpoint Management Patterns

### Periodic and Best-Model Checkpointing

```python
from pathlib import Path
import shutil


class CheckpointManager:
    """Manage training checkpoints with retention policies."""

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 5,
        keep_best: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best
        self.checkpoints: list[tuple[str, float, int]] = []  # (path, reward, iter)
        self.best_checkpoint: str | None = None
        self.best_reward = float("-inf")

    def save(
        self,
        algo: "Algorithm",
        iteration: int,
        reward: float,
    ) -> str:
        """Save checkpoint and manage retention."""
        # Save new checkpoint
        checkpoint_name = f"checkpoint_{iteration:06d}"
        checkpoint_path = str(self.checkpoint_dir / checkpoint_name)
        algo.save_checkpoint(checkpoint_path)

        self.checkpoints.append((checkpoint_path, reward, iteration))

        # Update best checkpoint
        if self.keep_best and reward > self.best_reward:
            self.best_reward = reward
            best_path = str(self.checkpoint_dir / "best")
            if Path(best_path).exists():
                shutil.rmtree(best_path)
            shutil.copytree(checkpoint_path, best_path)
            self.best_checkpoint = best_path

        # Enforce retention policy
        self._cleanup()

        return checkpoint_path

    def _cleanup(self) -> None:
        """Remove old checkpoints beyond retention limit."""
        while len(self.checkpoints) > self.keep_last_n:
            old_path, _, _ = self.checkpoints.pop(0)
            if Path(old_path).exists():
                shutil.rmtree(old_path)

    def get_best(self) -> str | None:
        """Return path to best checkpoint."""
        return self.best_checkpoint

    def get_latest(self) -> str | None:
        """Return path to most recent checkpoint."""
        if self.checkpoints:
            return self.checkpoints[-1][0]
        return None
```

### Training Resumption

```python
from ray.rllib.algorithms.ppo import PPO, PPOConfig
import ray


def resume_training(
    checkpoint_path: str,
    additional_iterations: int = 100,
) -> str:
    """Resume training from a checkpoint."""
    ray.init()

    try:
        # Restore algorithm from checkpoint
        algo = PPO.from_checkpoint(checkpoint_path)

        # Get iteration count from checkpoint
        state = algo.get_state()
        start_iteration = state.get("training_iteration", 0)

        checkpoint_manager = CheckpointManager(
            checkpoint_dir="./checkpoints_resumed",
            keep_last_n=3,
        )

        for i in range(additional_iterations):
            result = algo.train()
            iteration = start_iteration + i + 1
            reward = result.get("env_runners", {}).get("episode_return_mean", 0)

            print(f"Iteration {iteration}: reward={reward:.2f}")

            if (i + 1) % 10 == 0:
                checkpoint_manager.save(algo, iteration, reward)

        return checkpoint_manager.get_best()

    finally:
        algo.stop()
        ray.shutdown()
```

## Distributed Training Topology

### Single-Node Multi-GPU

```python
config = (
    PPOConfig()
    .environment("HalfCheetah-v4")
    .env_runners(
        num_env_runners=8,               # CPU workers for sampling
    )
    .learners(
        num_learners=2,                  # 2 GPU learners
        num_gpus_per_learner=1,          # 1 GPU each
        num_cpus_per_learner=1,
    )
    .training(
        train_batch_size_per_learner=2000,  # Per learner
        # Total batch = 2 * 2000 = 4000
    )
)
```

### Multi-Node Cluster

```python
# Assumes Ray cluster already initialized across nodes
config = (
    PPOConfig()
    .environment("HalfCheetah-v4")
    .env_runners(
        num_env_runners=32,              # Distributed across cluster
    )
    .learners(
        num_learners=4,                  # 4 learners on 4 GPUs
        num_gpus_per_learner=1,
        num_cpus_per_learner=4,
    )
    .training(
        train_batch_size_per_learner=4000,
        # Total batch = 4 * 4000 = 16000
    )
    .resources(
        num_cpus_for_main_process=1,
    )
)
```

### Resource Calculation

```python
def calculate_resources(config: "AlgorithmConfig") -> dict:
    """Calculate total resource requirements."""
    num_env_runners = config.num_env_runners
    num_learners = config.num_learners
    num_gpus_per_learner = config.num_gpus_per_learner
    num_cpus_per_learner = config.num_cpus_per_learner

    return {
        "total_cpus": (
            1  # Main process
            + num_env_runners  # Sampling workers
            + num_learners * num_cpus_per_learner  # Learner CPUs
        ),
        "total_gpus": num_learners * num_gpus_per_learner,
        "total_batch_size": (
            config.train_batch_size_per_learner * max(1, num_learners)
        ),
    }
```

## Ray Tune Integration

### Basic Hyperparameter Search

```python
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig


config = (
    PPOConfig()
    .environment("CartPole-v1")
    .env_runners(num_env_runners=2)
    .training(
        lr=tune.loguniform(1e-5, 1e-3),
        gamma=tune.uniform(0.95, 0.99),
        clip_param=tune.uniform(0.1, 0.3),
        num_epochs=tune.choice([5, 10, 20]),
        train_batch_size_per_learner=tune.choice([1000, 2000, 4000]),
    )
    .framework("torch")
)

tuner = tune.Tuner(
    "PPO",
    param_space=config,
    tune_config=tune.TuneConfig(
        metric="env_runners/episode_return_mean",
        mode="max",
        num_samples=20,
    ),
    run_config=tune.RunConfig(
        stop={"training_iteration": 50},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=10,
            num_to_keep=2,
        ),
    ),
)

results = tuner.fit()
best_result = results.get_best_result()
print(f"Best config: {best_result.config}")
print(f"Best reward: {best_result.metrics['env_runners/episode_return_mean']}")
```

### Population-Based Training

```python
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.ppo import PPOConfig
import random


def explore_fn(config: dict) -> dict:
    """Post-process perturbed config for validity."""
    # Ensure batch size constraints
    if config.get("train_batch_size_per_learner", 0) < config.get(
        "minibatch_size", 128
    ):
        config["train_batch_size_per_learner"] = config["minibatch_size"] * 4
    return config


pbt_scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=5,
    hyperparam_mutations={
        "lr": lambda: random.uniform(1e-5, 1e-3),
        "clip_param": lambda: random.uniform(0.1, 0.3),
        "num_epochs": lambda: random.randint(3, 20),
        "train_batch_size_per_learner": [1000, 2000, 4000, 8000],
    },
    custom_explore_fn=explore_fn,
)

config = (
    PPOConfig()
    .environment("HalfCheetah-v4")
    .env_runners(num_env_runners=4)
    .training(
        lr=1e-4,
        clip_param=0.2,
        num_epochs=10,
        train_batch_size_per_learner=4000,
    )
    .framework("torch")
)

tuner = tune.Tuner(
    "PPO",
    param_space=config,
    tune_config=tune.TuneConfig(
        metric="env_runners/episode_return_mean",
        mode="max",
        scheduler=pbt_scheduler,
        num_samples=8,  # Population size
    ),
    run_config=tune.RunConfig(
        stop={"training_iteration": 100},
    ),
)

results = tuner.fit()
```

### ASHA Scheduler for Early Stopping

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.ppo import PPOConfig


asha_scheduler = ASHAScheduler(
    time_attr="training_iteration",
    metric="env_runners/episode_return_mean",
    mode="max",
    max_t=100,           # Max iterations
    grace_period=10,     # Min iterations before stopping
    reduction_factor=3,  # Aggressive pruning
)

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .env_runners(num_env_runners=2)
    .training(
        lr=tune.loguniform(1e-5, 1e-3),
        gamma=tune.uniform(0.9, 0.999),
        train_batch_size_per_learner=tune.choice([1000, 2000, 4000]),
    )
    .framework("torch")
)

tuner = tune.Tuner(
    "PPO",
    param_space=config,
    tune_config=tune.TuneConfig(
        scheduler=asha_scheduler,
        num_samples=50,
    ),
    run_config=tune.RunConfig(
        stop={"training_iteration": 100},
    ),
)

results = tuner.fit()
```

## Production Training Template

Complete production-ready training script combining all patterns:

```python
#!/usr/bin/env python3
"""Production RLlib training script with full lifecycle management."""

import argparse
import logging
from pathlib import Path
from typing import Any

import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ProductionCallback(RLlibCallback):
    """Production monitoring callback."""

    def __init__(self):
        super().__init__()
        self.best_reward = float("-inf")

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict[str, Any],
        **kwargs,
    ) -> None:
        reward = result.get("env_runners", {}).get("episode_return_mean", 0)
        if reward > self.best_reward:
            self.best_reward = reward
            result["is_best"] = True


def train(
    env_name: str,
    num_iterations: int,
    checkpoint_dir: str,
    num_env_runners: int,
    num_gpus: int,
    target_reward: float | None,
    resume_from: str | None,
) -> str:
    """Run production training."""
    ray.init()
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    try:
        if resume_from:
            logger.info(f"Resuming from {resume_from}")
            algo = PPO.from_checkpoint(resume_from)
            start_iter = algo.get_state().get("training_iteration", 0)
        else:
            config = (
                PPOConfig()
                .environment(env_name)
                .env_runners(num_env_runners=num_env_runners)
                .learners(
                    num_learners=num_gpus if num_gpus > 0 else 0,
                    num_gpus_per_learner=1 if num_gpus > 0 else 0,
                )
                .training(
                    gamma=0.99,
                    lr=3e-4,
                    train_batch_size_per_learner=4000,
                    num_epochs=10,
                    minibatch_size=128,
                    grad_clip=0.5,
                )
                .callbacks(ProductionCallback)
                .framework("torch")
            )
            algo = config.build()
            start_iter = 0

        best_reward = float("-inf")
        best_checkpoint = None

        for i in range(num_iterations):
            iteration = start_iter + i + 1
            result = algo.train()
            reward = result.get("env_runners", {}).get("episode_return_mean", 0)

            logger.info(f"Iteration {iteration}: reward={reward:.2f}")

            # Save checkpoints
            if (i + 1) % 10 == 0:
                path = algo.save_checkpoint(str(checkpoint_path / f"iter_{iteration}"))
                logger.info(f"Checkpoint saved: {path}")

            if reward > best_reward:
                best_reward = reward
                best_checkpoint = algo.save_checkpoint(
                    str(checkpoint_path / "best")
                )
                logger.info(f"New best: {best_reward:.2f}")

            if target_reward and reward >= target_reward:
                logger.info(f"Target reward {target_reward} reached!")
                break

        return best_checkpoint

    finally:
        algo.stop()
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="RLlib Production Training")
    parser.add_argument("--env", default="CartPole-v1", help="Environment name")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--num-env-runners", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--target-reward", type=float, default=None)
    parser.add_argument("--resume-from", default=None)

    args = parser.parse_args()

    best = train(
        env_name=args.env,
        num_iterations=args.iterations,
        checkpoint_dir=args.checkpoint_dir,
        num_env_runners=args.num_env_runners,
        num_gpus=args.num_gpus,
        target_reward=args.target_reward,
        resume_from=args.resume_from,
    )

    print(f"Training complete. Best checkpoint: {best}")


if __name__ == "__main__":
    main()
```

**Usage:**

```bash
# Basic training
python train.py --env CartPole-v1 --iterations 100

# GPU training with target reward
python train.py --env HalfCheetah-v4 --num-gpus 2 --target-reward 5000

# Resume from checkpoint
python train.py --resume-from ./checkpoints/best --iterations 50
```
