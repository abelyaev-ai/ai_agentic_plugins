---
name: rllib-advanced-techniques
description: >
  This skill should be used when the user asks about "offline RL",
  "behavioral cloning", "hierarchical RL", "meta-RL",
  "curiosity-driven exploration", or implements advanced RLlib techniques.
---

## Purpose

Provide implementation guidance for advanced reinforcement learning techniques available through RLlib's new API stack. This skill covers offline RL algorithms (CQL, IQL, MARWIL, BC), hierarchical policy architectures, meta-learning workflows, curiosity-driven exploration mechanisms, and custom RLModule architectures for specialized neural network designs including attention mechanisms and graph neural networks.

## Prerequisites

Before implementing advanced RL techniques, resolve up-to-date documentation using context7:

1. Call `resolve-library-id` with library name "ray rllib" to obtain the Context7-compatible library ID
2. Call `query-docs` with queries for the specific technique:
   - Offline RL: "CQL IQL MARWIL behavioral cloning offline data configuration"
   - Hierarchical RL: "hierarchical environment options framework temporal abstraction"
   - Meta-RL: "meta-learning task distribution few-shot adaptation"
   - Exploration: "curiosity intrinsic motivation RND ICM count-based exploration"
   - Custom RLModule: "RLModule custom architecture forward method Catalog"

Required packages:
- `ray[rllib]>=2.53` - RLlib with new API stack support
- `torch>=2.0` - PyTorch backend (TensorFlow not supported on new API stack)
- `gymnasium>=0.28` - Environment interface
- `numpy` - Numerical operations

## Core Workflow

### Offline RL: Training from Static Datasets

Offline RL enables training policies from pre-collected datasets without environment interaction. RLlib supports multiple offline algorithms through the new API stack, with IQL and MARWIL being the primary options.

**Dataset Integration**

Offline datasets must conform to RLlib's expected schema. Supported formats include JSON-lines and Parquet files with the following structure:

```python
# Required columns in offline dataset
SCHEMA = {
    "obs": "observation at timestep t",
    "actions": "action taken at timestep t",
    "rewards": "reward received after action",
    "new_obs": "observation at timestep t+1",
    "terminateds": "whether episode terminated",
    "truncateds": "whether episode was truncated",
}
```

Prepare datasets using Ray Data for efficient distributed loading:

```python
import ray
from ray import data as ray_data

# Load Parquet dataset
dataset = ray_data.read_parquet("s3://bucket/offline_data/")

# Verify schema compliance
sample = dataset.take(1)[0]
required_keys = ["obs", "actions", "rewards", "new_obs", "terminateds", "truncateds"]
assert all(key in sample for key in required_keys), "Missing required columns"
```

**IQL Algorithm Configuration**

Implicit Q-Learning (IQL) avoids querying out-of-distribution actions by using expectile regression. Configure IQL for offline training:

```python
from ray.rllib.algorithms.iql import IQLConfig

config = (
    IQLConfig()
    .environment(env="Pendulum-v1")
    .training(
        # IQL-specific hyperparameters
        beta=0.1,  # Temperature for actor loss (lower = more conservative)
        expectile=0.8,  # Expectile value (0.5 = mean, higher = optimistic)
        twin_q=True,  # Use twin Q-networks to reduce overestimation
        # Learning rates for each network
        actor_lr=3e-4,
        critic_lr=3e-4,
        value_lr=3e-4,
        # Target network updates
        tau=0.005,  # Polyak averaging coefficient
        gamma=0.99,
    )
    .offline_data(
        input_="/path/to/offline/dataset",
        # Dataset configuration
        input_read_method="read_parquet",  # or "read_json"
        input_read_method_kwargs={"parallelism": 100},
    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=1,
    )
)

algo = config.build()
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: loss={result['info']['learner']['default_policy']['total_loss']}")
```

**Behavioral Cloning Setup**

Behavioral cloning (BC) treats offline RL as supervised learning, directly imitating the behavior policy. MARWIL with `beta=0` provides BC functionality:

```python
from ray.rllib.algorithms.marwil import MARWILConfig

bc_config = (
    MARWILConfig()
    .environment(env="CartPole-v1")
    .training(
        beta=0.0,  # Set to 0 for pure behavioral cloning
        gamma=0.99,
        lr=1e-4,
    )
    .offline_data(
        input_="/path/to/expert/demonstrations",
    )
)

bc_algo = bc_config.build()
```

**Offline-to-Online Fine-tuning**

Transition from offline pre-training to online fine-tuning by loading a checkpoint and reconfiguring for environment interaction:

```python
from ray.rllib.algorithms.sac import SACConfig

# Load offline-trained checkpoint
checkpoint_path = "/path/to/offline/checkpoint"

# Configure for online fine-tuning
online_config = (
    SACConfig()
    .environment(env="Pendulum-v1")
    .training(
        initial_alpha=0.1,  # Start with lower exploration
        lr=1e-5,  # Lower learning rate for fine-tuning
    )
    .env_runners(
        num_env_runners=4,
        num_envs_per_env_runner=1,
    )
)

# Build and restore from offline checkpoint
algo = online_config.build()
algo.restore(checkpoint_path)

# Continue training online
for i in range(100):
    result = algo.train()
```

### Hierarchical RL: Temporal Abstraction and Options

Hierarchical RL decomposes complex tasks into sub-tasks using temporal abstraction. Implement hierarchical policies using RLlib's multi-agent framework where a manager policy selects sub-goals and worker policies execute primitive actions.

**Option Framework Pattern**

Model the options framework using a two-level hierarchy:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class HierarchicalEnv(MultiAgentEnv):
    """Hierarchical environment with manager and worker agents."""

    def __init__(self, config):
        super().__init__()
        self.base_env = gym.make(config.get("base_env", "MountainCarContinuous-v0"))
        self.num_options = config.get("num_options", 4)
        self.option_duration = config.get("option_duration", 10)

        # Manager observes full state, selects option
        self._agent_ids = {"manager", "worker"}
        self.observation_space = {
            "manager": self.base_env.observation_space,
            "worker": spaces.Dict({
                "obs": self.base_env.observation_space,
                "option": spaces.Discrete(self.num_options),
            }),
        }
        self.action_space = {
            "manager": spaces.Discrete(self.num_options),
            "worker": self.base_env.action_space,
        }

        self._current_option = None
        self._steps_in_option = 0

    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed)
        self._current_option = None
        self._steps_in_option = 0
        # Manager acts first
        return {"manager": obs}, {"manager": info}

    def step(self, action_dict):
        if "manager" in action_dict:
            # Manager selected new option
            self._current_option = action_dict["manager"]
            self._steps_in_option = 0
            obs, _ = self.base_env.unwrapped.state, {}
            return (
                {"worker": {"obs": obs, "option": self._current_option}},
                {},
                {"__all__": False},
                {"__all__": False},
                {},
            )

        # Worker executes primitive action
        obs, reward, terminated, truncated, info = self.base_env.step(action_dict["worker"])
        self._steps_in_option += 1

        # Check if option should terminate
        option_done = self._steps_in_option >= self.option_duration or terminated or truncated

        if option_done and not (terminated or truncated):
            # Return control to manager
            return (
                {"manager": obs},
                {"manager": reward * self._steps_in_option},  # Accumulated reward
                {"__all__": False},
                {"__all__": False},
                {},
            )

        return (
            {"worker": {"obs": obs, "option": self._current_option}},
            {"worker": reward},
            {"__all__": terminated},
            {"__all__": truncated},
            {"worker": info},
        )
```

**Hierarchical Policy Configuration**

Configure separate policies for manager and worker:

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(
        env=HierarchicalEnv,
        env_config={"base_env": "MountainCarContinuous-v0", "num_options": 4},
    )
    .multi_agent(
        policies={"manager", "worker"},
        policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id,
    )
    .training(
        gamma=0.99,
        lr=3e-4,
    )
)
```

### Meta-RL: Learning to Learn

Meta-RL trains policies that adapt quickly to new tasks. Implement meta-learning by constructing task distributions and training with episodic adaptation.

**Task Distribution Setup**

Create a task distribution by parameterizing environment dynamics:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MetaLearningEnv(gym.Env):
    """Environment with task-parameterized dynamics."""

    def __init__(self, config=None):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Task parameters sampled at episode start
        self._goal = None
        self._dynamics_scale = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Sample new task
        self._goal = self.np_random.uniform(-1, 1, size=(2,))
        self._dynamics_scale = self.np_random.uniform(0.5, 2.0)
        self._state = self.np_random.uniform(-0.1, 0.1, size=(2,))
        return self._get_obs(), {}

    def _get_obs(self):
        # Include goal in observation for context
        return np.concatenate([self._state, self._goal]).astype(np.float32)

    def step(self, action):
        # Task-dependent dynamics
        self._state = self._state + action[0] * self._dynamics_scale * 0.1
        self._state = np.clip(self._state, -1, 1)

        reward = -np.linalg.norm(self._state - self._goal)
        terminated = np.linalg.norm(self._state - self._goal) < 0.1
        return self._get_obs(), reward, terminated, False, {}
```

**Meta-Learning Training Loop**

Structure training to expose the agent to diverse tasks:

```python
from ray.rllib.algorithms.ppo import PPOConfig

meta_config = (
    PPOConfig()
    .environment(
        env=MetaLearningEnv,
    )
    .training(
        gamma=0.99,
        lr=3e-4,
        # Longer rollouts for within-episode adaptation
        train_batch_size_per_learner=4000,
    )
    .env_runners(
        num_env_runners=8,
        rollout_fragment_length=200,  # Capture full adaptation episodes
    )
    .rl_module(
        model_config={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
        },
    )
)

algo = meta_config.build()
# Train across many task samples
for i in range(1000):
    result = algo.train()
```

### Exploration Strategies: Intrinsic Motivation

Implement curiosity-driven exploration to improve sample efficiency in sparse-reward environments.

**Intrinsic Curiosity Module (ICM) Pattern**

Add intrinsic rewards based on prediction error:

```python
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np

class CuriosityCallback(DefaultCallbacks):
    """Add intrinsic curiosity reward based on state prediction error."""

    def __init__(self):
        super().__init__()
        self._predictor = None  # Initialize predictor model

    def on_episode_step(
        self,
        *,
        episode,
        env_runner,
        env,
        env_index,
        rl_module,
        **kwargs,
    ):
        # Get current transition
        obs = episode.get_observations()[-1]
        prev_obs = episode.get_observations()[-2] if len(episode.get_observations()) > 1 else obs
        action = episode.get_actions()[-1]

        # Compute prediction error as intrinsic reward
        # (simplified - actual implementation needs trained predictor)
        intrinsic_reward = self._compute_prediction_error(prev_obs, action, obs)

        # Add to extrinsic reward
        episode.custom_metrics["intrinsic_reward"] = intrinsic_reward

    def _compute_prediction_error(self, prev_obs, action, obs):
        # Placeholder for actual prediction model
        return np.random.uniform(0, 0.1)


config = (
    PPOConfig()
    .environment(env="MontezumaRevenge-v4")
    .callbacks(CuriosityCallback)
    .training(
        gamma=0.999,  # Higher gamma for exploration
        lr=2.5e-4,
    )
)
```

**Exploration vs Exploitation Scheduling**

Implement exploration coefficient annealing:

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(env="CartPole-v1")
    .exploration(
        exploration_config={
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.01,
            "epsilon_timesteps": 100000,
        }
    )
)
```

### Custom RLModule: Advanced Architectures

Extend RLModule to implement custom neural network architectures including attention mechanisms and graph neural networks.

**Basic Custom RLModule Structure**

```python
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.columns import Columns
import torch
import torch.nn as nn

class CustomRLModule(TorchRLModule):
    """Custom RLModule with specialized architecture."""

    def setup(self):
        # Define network architecture
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.n  # Discrete actions

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)

    def _forward_inference(self, batch, **kwargs):
        obs = batch[Columns.OBS]
        features = self.encoder(obs)
        logits = self.policy_head(features)
        return {Columns.ACTION_DIST_INPUTS: logits}

    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    def _forward_train(self, batch, **kwargs):
        obs = batch[Columns.OBS]
        features = self.encoder(obs)
        logits = self.policy_head(features)
        values = self.value_head(features)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values.squeeze(-1),
        }
```

**Attention Mechanism Integration**

```python
class AttentionRLModule(TorchRLModule):
    """RLModule with self-attention for sequential observations."""

    def setup(self):
        obs_dim = self.observation_space.shape[-1]
        self.embedding = nn.Linear(obs_dim, 128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.policy_head = nn.Linear(128, self.action_space.n)
        self.value_head = nn.Linear(128, 1)

    def _forward_train(self, batch, **kwargs):
        obs = batch[Columns.OBS]  # Shape: (batch, seq_len, obs_dim)
        embedded = self.embedding(obs)
        attended, _ = self.attention(embedded, embedded, embedded)
        pooled = attended.mean(dim=1)  # Global average pooling
        return {
            Columns.ACTION_DIST_INPUTS: self.policy_head(pooled),
            Columns.VF_PREDS: self.value_head(pooled).squeeze(-1),
        }
```

**Registering Custom RLModule**

```python
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

config = (
    PPOConfig()
    .environment(env="CartPole-v1")
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=CustomRLModule,
        ),
    )
)
```

## Common Pitfalls

- **Offline data distribution mismatch**: Offline algorithms assume the dataset covers the state-action space relevant to the target policy. Datasets collected from random or suboptimal policies may not support learning high-performing policies. Always analyze dataset coverage before training.

- **Ignoring IQL/CQL conservatism hyperparameters**: Setting `beta` too high in IQL causes excessive conservatism; too low leads to overestimation. Start with default values (beta=0.1, expectile=0.8) and tune based on dataset quality.

- **Hierarchical reward assignment errors**: In hierarchical setups, ensure rewards flow correctly between manager and worker policies. Accumulated rewards for managers should reflect the value of sub-goal selection, not individual primitive actions.

- **Custom RLModule forward method mismatches**: The three forward methods (`_forward_inference`, `_forward_exploration`, `_forward_train`) serve different purposes. Training requires value predictions; inference does not. Implement all three methods with appropriate outputs.

- **Exploration intrinsic reward scaling**: Intrinsic rewards from curiosity modules can dominate extrinsic rewards if not properly scaled. Use adaptive scaling or clip intrinsic rewards to maintain balance.

## Additional Resources

For detailed reference documentation including offline data format specifications, algorithm hyperparameter tuning guides, hierarchical architecture examples, and custom RLModule implementation patterns, see `references/advanced-rl.md`.

Key topics covered in the reference guide:
- Offline RL data format requirements and preparation pipelines
- CQL and IQL hyperparameter sensitivity analysis
- Hierarchical policy architecture design patterns
- Meta-learning algorithm comparison (MAML, RL2, PEARL)
- Exploration strategy selection criteria
- Custom RLModule patterns for attention and graph networks
- Reward model integration for RLHF
- Model-based RL approaches with RLlib
