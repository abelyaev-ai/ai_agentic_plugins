# Advanced RL Reference Guide

This reference provides detailed implementation guidance for advanced reinforcement learning techniques in RLlib, including offline RL, hierarchical policies, meta-learning, exploration strategies, and custom RLModule architectures.

## Offline RL Data Format Requirements

### Schema Specification

RLlib's offline data pipeline expects datasets conforming to a strict schema. Each record represents a single transition in the environment.

**Required Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `obs` | array | Observation at timestep t |
| `actions` | array/scalar | Action taken at timestep t |
| `rewards` | float | Scalar reward received |
| `new_obs` | array | Observation at timestep t+1 |
| `terminateds` | bool | Episode termination flag |
| `truncateds` | bool | Episode truncation flag |

**Optional Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `infos` | dict | Additional transition metadata |
| `action_prob` | float | Probability of action under behavior policy |
| `action_logp` | float | Log probability of action |
| `eps_id` | int | Episode identifier for grouping |
| `agent_id` | str | Agent identifier (multi-agent) |

### Data Preparation Pipeline

Prepare offline datasets using Ray Data for distributed processing:

```python
import ray
from ray import data as ray_data
import numpy as np
from typing import Dict, Any
import json

def validate_transition(record: Dict[str, Any]) -> bool:
    """Validate a single transition record."""
    required = ["obs", "actions", "rewards", "new_obs", "terminateds", "truncateds"]
    return all(key in record for key in required)

def normalize_observations(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalize observations to zero mean, unit variance."""
    obs = batch["obs"]
    new_obs = batch["new_obs"]

    # Compute statistics (in practice, use pre-computed values)
    mean = obs.mean(axis=0)
    std = obs.std(axis=0) + 1e-8

    batch["obs"] = (obs - mean) / std
    batch["new_obs"] = (new_obs - mean) / std
    return batch

def prepare_offline_dataset(
    input_path: str,
    output_path: str,
    num_workers: int = 4,
) -> None:
    """Prepare and validate offline dataset for RLlib training."""
    ray.init(ignore_reinit_error=True)

    # Load raw data
    if input_path.endswith(".json") or input_path.endswith(".jsonl"):
        dataset = ray_data.read_json(input_path)
    elif input_path.endswith(".parquet"):
        dataset = ray_data.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported format: {input_path}")

    # Validate schema
    sample = dataset.take(1)[0]
    if not validate_transition(sample):
        raise ValueError("Dataset missing required columns")

    # Process and normalize
    dataset = dataset.map_batches(
        normalize_observations,
        batch_format="numpy",
    )

    # Write processed dataset
    dataset.write_parquet(output_path)
    print(f"Processed {dataset.count()} transitions to {output_path}")
```

### JSON-Lines Format Example

```json
{"obs": [0.1, -0.2, 0.5], "actions": 1, "rewards": 0.0, "new_obs": [0.15, -0.18, 0.52], "terminateds": false, "truncateds": false}
{"obs": [0.15, -0.18, 0.52], "actions": 0, "rewards": 1.0, "new_obs": [0.2, -0.1, 0.6], "terminateds": true, "truncateds": false}
```

### Parquet Format Considerations

Parquet provides superior performance for large datasets:

- Use columnar compression (snappy or zstd)
- Partition by episode ID for efficient sampling
- Store observations as fixed-size arrays, not lists
- Use appropriate data types (float32 for observations, int32 for discrete actions)

```python
import pyarrow as pa
import pyarrow.parquet as pq

schema = pa.schema([
    ("obs", pa.list_(pa.float32(), list_size=4)),
    ("actions", pa.int32()),
    ("rewards", pa.float32()),
    ("new_obs", pa.list_(pa.float32(), list_size=4)),
    ("terminateds", pa.bool_()),
    ("truncateds", pa.bool_()),
    ("eps_id", pa.int64()),
])

# Write with appropriate settings
pq.write_table(
    table,
    "offline_data.parquet",
    compression="snappy",
    row_group_size=10000,
)
```

## CQL and IQL Hyperparameter Tuning

### Conservative Q-Learning (CQL) Parameters

CQL adds a conservative penalty to prevent overestimation of out-of-distribution actions.

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `cql_alpha` | 1.0 | [0.1, 10.0] | Conservative penalty weight. Higher = more conservative |
| `cql_clip_diff_min` | -inf | [-10, 0] | Minimum Q-value difference to clip |
| `cql_clip_diff_max` | inf | [0, 10] | Maximum Q-value difference to clip |
| `num_actions_sampled` | 10 | [1, 50] | Actions sampled for CQL loss |
| `lagrangian` | False | bool | Use Lagrangian constraint for alpha |
| `lagrangian_thresh` | 10.0 | [1, 100] | Target value for Lagrangian |

**Tuning Strategy:**

1. Start with default `cql_alpha=1.0`
2. If policy is too conservative (low returns), decrease alpha
3. If policy diverges or shows overestimation, increase alpha
4. Enable Lagrangian mode for automatic alpha tuning on diverse datasets

```python
from ray.rllib.algorithms.cql import CQLConfig

# Conservative configuration for narrow dataset
conservative_config = (
    CQLConfig()
    .training(
        cql_alpha=5.0,
        num_actions_sampled=20,
    )
)

# Adaptive configuration using Lagrangian
adaptive_config = (
    CQLConfig()
    .training(
        lagrangian=True,
        lagrangian_thresh=5.0,
    )
)
```

### Implicit Q-Learning (IQL) Parameters

IQL avoids OOD actions through expectile regression, not explicit penalties.

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `beta` | 0.1 | [0.01, 1.0] | Actor temperature. Lower = more conservative |
| `expectile` | 0.8 | [0.5, 0.99] | Expectile for value regression. Higher = more optimistic |
| `twin_q` | True | bool | Use twin Q-networks |
| `actor_lr` | 3e-4 | [1e-5, 1e-3] | Actor learning rate |
| `critic_lr` | 3e-4 | [1e-5, 1e-3] | Critic learning rate |
| `value_lr` | 3e-4 | [1e-5, 1e-3] | Value network learning rate |
| `tau` | 0.005 | [0.001, 0.1] | Target network Polyak coefficient |

**Tuning Strategy:**

1. Start with defaults: `beta=0.1`, `expectile=0.8`
2. For high-quality expert data, increase `expectile` toward 0.9
3. For mixed-quality data, decrease `expectile` toward 0.7
4. If training is unstable, decrease learning rates uniformly
5. `beta` controls policy extraction: lower values stick closer to data

```python
from ray.rllib.algorithms.iql import IQLConfig

# Configuration for expert demonstrations
expert_config = (
    IQLConfig()
    .training(
        beta=0.3,  # More aggressive extraction
        expectile=0.9,  # Trust high-value actions
        twin_q=True,
        actor_lr=1e-4,
        critic_lr=3e-4,
        value_lr=3e-4,
    )
)

# Configuration for suboptimal data
suboptimal_config = (
    IQLConfig()
    .training(
        beta=0.05,  # Very conservative
        expectile=0.7,  # Pessimistic value estimates
        tau=0.001,  # Slow target updates
    )
)
```

### Behavioral Cloning with MARWIL

MARWIL with `beta=0` provides behavioral cloning:

```python
from ray.rllib.algorithms.marwil import MARWILConfig

bc_config = (
    MARWILConfig()
    .training(
        beta=0.0,  # Pure BC (no advantage weighting)
        lr=1e-4,
        train_batch_size_per_learner=256,
    )
    .offline_data(
        input_="/path/to/demonstrations",
    )
)

# MARWIL with advantage weighting (beta > 0)
marwil_config = (
    MARWILConfig()
    .training(
        beta=1.0,  # Weight by exponential advantage
        vf_coeff=1.0,
        lr=3e-4,
    )
)
```

## Hierarchical Policy Architecture Examples

### Two-Level Hierarchy with Options

Implement a manager-worker hierarchy where the manager selects temporal abstractions (options) and workers execute primitive actions.

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Dict, Tuple, Any, Optional

class TwoLevelHierarchicalEnv(MultiAgentEnv):
    """
    Two-level hierarchical environment implementing the options framework.

    Manager: Observes state, selects option (sub-goal)
    Worker: Observes state + current option, executes primitive actions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}

        self.base_env_name = config.get("base_env", "LunarLander-v3")
        self.base_env = gym.make(self.base_env_name)
        self.num_options = config.get("num_options", 4)
        self.max_option_length = config.get("max_option_length", 25)
        self.option_termination_prob = config.get("option_termination_prob", 0.1)

        # Agent IDs
        self._agent_ids = {"manager", "worker"}

        # Observation spaces
        base_obs_space = self.base_env.observation_space
        self.observation_space = {
            "manager": base_obs_space,
            "worker": spaces.Dict({
                "obs": base_obs_space,
                "option": spaces.Discrete(self.num_options),
                "option_steps": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }),
        }

        # Action spaces
        self.action_space = {
            "manager": spaces.Discrete(self.num_options),
            "worker": self.base_env.action_space,
        }

        # State tracking
        self._current_option: Optional[int] = None
        self._option_steps: int = 0
        self._accumulated_reward: float = 0.0
        self._base_obs: Optional[np.ndarray] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._base_obs, info = self.base_env.reset(seed=seed)
        self._current_option = None
        self._option_steps = 0
        self._accumulated_reward = 0.0

        # Manager acts first to select initial option
        return {"manager": self._base_obs}, {"manager": info}

    def step(
        self,
        action_dict: Dict[str, Any],
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        if "manager" in action_dict:
            return self._manager_step(action_dict["manager"])
        else:
            return self._worker_step(action_dict["worker"])

    def _manager_step(
        self,
        option: int,
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Manager selects a new option."""
        self._current_option = option
        self._option_steps = 0
        self._accumulated_reward = 0.0

        worker_obs = {
            "obs": self._base_obs,
            "option": self._current_option,
            "option_steps": np.array([0.0], dtype=np.float32),
        }

        return (
            {"worker": worker_obs},
            {},  # No reward for manager yet
            {"__all__": False},
            {"__all__": False},
            {},
        )

    def _worker_step(
        self,
        action: Any,
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Worker executes primitive action."""
        self._base_obs, reward, terminated, truncated, info = self.base_env.step(action)
        self._option_steps += 1
        self._accumulated_reward += reward

        # Check option termination conditions
        option_done = (
            terminated or
            truncated or
            self._option_steps >= self.max_option_length or
            np.random.random() < self.option_termination_prob
        )

        if option_done and not (terminated or truncated):
            # Option terminates, return to manager
            return (
                {"manager": self._base_obs},
                {"manager": self._accumulated_reward},
                {"__all__": False},
                {"__all__": False},
                {"manager": {"option_length": self._option_steps}},
            )

        if terminated or truncated:
            # Episode ends
            return (
                {"manager": self._base_obs, "worker": self._get_worker_obs()},
                {"manager": self._accumulated_reward, "worker": reward},
                {"__all__": terminated},
                {"__all__": truncated},
                {},
            )

        # Continue current option
        return (
            {"worker": self._get_worker_obs()},
            {"worker": reward},
            {"__all__": False},
            {"__all__": False},
            {"worker": info},
        )

    def _get_worker_obs(self) -> Dict[str, Any]:
        return {
            "obs": self._base_obs,
            "option": self._current_option,
            "option_steps": np.array(
                [self._option_steps / self.max_option_length],
                dtype=np.float32,
            ),
        }
```

### Feudal Networks Pattern

Implement feudal RL with directional sub-goals:

```python
import torch
import torch.nn as nn
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.columns import Columns

class FeudalManagerModule(TorchRLModule):
    """Manager network that outputs directional goals in embedding space."""

    def setup(self):
        obs_dim = self.observation_space.shape[0]
        self.goal_dim = 64
        self.horizon = 10  # Manager acts every 10 steps

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Output goal direction (unit vector)
        self.goal_head = nn.Sequential(
            nn.Linear(256, self.goal_dim),
        )

        self.value_head = nn.Linear(256, 1)

    def _forward_train(self, batch, **kwargs):
        obs = batch[Columns.OBS]
        features = self.encoder(obs)

        # Normalize goal to unit vector
        goal = self.goal_head(features)
        goal = goal / (goal.norm(dim=-1, keepdim=True) + 1e-8)

        values = self.value_head(features)

        return {
            "goal": goal,
            Columns.VF_PREDS: values.squeeze(-1),
        }


class FeudalWorkerModule(TorchRLModule):
    """Worker network conditioned on manager's goal."""

    def setup(self):
        obs_dim = self.observation_space["obs"].shape[0]
        goal_dim = 64
        action_dim = self.action_space.n

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
        )

        # Goal embedding
        self.goal_encoder = nn.Linear(goal_dim, 256)

        # Combined processing
        self.combined = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)

    def _forward_train(self, batch, **kwargs):
        obs = batch[Columns.OBS]["obs"]
        goal = batch[Columns.OBS]["goal"]

        obs_features = self.encoder(obs)
        goal_features = self.goal_encoder(goal)

        combined = torch.cat([obs_features, goal_features], dim=-1)
        features = self.combined(combined)

        return {
            Columns.ACTION_DIST_INPUTS: self.policy_head(features),
            Columns.VF_PREDS: self.value_head(features).squeeze(-1),
        }
```

## Meta-Learning Algorithm Comparison

### Algorithm Overview

| Algorithm | Key Idea | Adaptation | Compute Cost |
|-----------|----------|------------|--------------|
| MAML | Gradient-based meta-learning | Few gradient steps | High (second-order) |
| RL2 | RNN-based task inference | In-context (no gradients) | Low |
| PEARL | Probabilistic context encoder | Latent inference | Medium |
| MAML++ | Stabilized MAML | Few gradient steps | Medium |
| ProMP | First-order MAML approximation | Few gradient steps | Low |

### RL2 Implementation Pattern

RL2 uses recurrent networks to infer task identity from experience:

```python
import torch
import torch.nn as nn
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.columns import Columns

class RL2Module(TorchRLModule):
    """
    RL2: Fast Reinforcement Learning via Slow Reinforcement Learning

    Uses GRU to process episode history and infer task context.
    """

    def setup(self):
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.n

        # Input: obs + prev_action + prev_reward
        input_dim = obs_dim + action_dim + 1
        hidden_dim = 256

        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self._hidden = None

    def _forward_train(self, batch, **kwargs):
        # Batch contains sequences
        obs = batch[Columns.OBS]  # (batch, seq, obs_dim)
        prev_actions = batch.get("prev_actions")  # (batch, seq, action_dim)
        prev_rewards = batch.get("prev_rewards")  # (batch, seq, 1)

        # Concatenate inputs
        inputs = torch.cat([obs, prev_actions, prev_rewards], dim=-1)

        # Process through GRU
        outputs, self._hidden = self.gru(inputs, self._hidden)

        return {
            Columns.ACTION_DIST_INPUTS: self.policy_head(outputs),
            Columns.VF_PREDS: self.value_head(outputs).squeeze(-1),
        }

    def reset_hidden(self, batch_size: int = 1):
        """Reset hidden state at episode boundaries."""
        self._hidden = torch.zeros(1, batch_size, 256)
```

### PEARL Context Encoder

```python
class PEARLEncoder(nn.Module):
    """
    Probabilistic Embeddings for Actor-critic RL (PEARL)

    Encodes task context from transitions into a latent distribution.
    """

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 64):
        super().__init__()

        # Transition encoder
        input_dim = obs_dim * 2 + action_dim + 1  # (s, a, r, s')
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Output mean and variance of latent
        self.mean_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode context transitions into latent distribution."""
        # (batch, context_size, dim)
        inputs = torch.cat([obs, actions, rewards, next_obs], dim=-1)

        # Encode each transition
        encoded = self.encoder(inputs)

        # Aggregate across context (mean pooling)
        aggregated = encoded.mean(dim=1)

        mean = self.mean_head(aggregated)
        logvar = self.logvar_head(aggregated)

        return mean, logvar

    def sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterized sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
```

## Exploration Strategy Selection Guide

### Strategy Comparison

| Strategy | Best For | Sample Efficiency | Implementation Complexity |
|----------|----------|-------------------|--------------------------|
| Epsilon-greedy | Simple discrete actions | Low | Low |
| Boltzmann/Softmax | Smooth action selection | Medium | Low |
| UCB | Bandit-like problems | High | Medium |
| ICM | Sparse rewards, curiosity | High | High |
| RND | Large state spaces | High | Medium |
| Count-based | Tabular/small discrete | Very High | Medium |
| NoisyNet | Continuous control | Medium | Medium |

### Intrinsic Curiosity Module (ICM) Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class ICMModel(nn.Module):
    """
    Intrinsic Curiosity Module

    Computes intrinsic reward from prediction error of a forward dynamics model.
    """

    def __init__(self, obs_dim: int, action_dim: int, feature_dim: int = 256):
        super().__init__()

        # Feature encoder (shared)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        # Forward model: predicts next features from current features + action
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        # Inverse model: predicts action from current and next features
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute forward and inverse model outputs."""
        # Encode observations
        features = self.encoder(obs)
        next_features = self.encoder(next_obs)

        # Forward model prediction
        action_onehot = F.one_hot(action, num_classes=self.inverse_model[-1].out_features).float()
        forward_input = torch.cat([features, action_onehot], dim=-1)
        predicted_next_features = self.forward_model(forward_input)

        # Inverse model prediction
        inverse_input = torch.cat([features, next_features], dim=-1)
        predicted_action_logits = self.inverse_model(inverse_input)

        return predicted_next_features, next_features, predicted_action_logits

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        eta: float = 0.01,
    ) -> torch.Tensor:
        """Compute intrinsic reward from forward model prediction error."""
        with torch.no_grad():
            pred_next, actual_next, _ = self.forward(obs, next_obs, action)
            intrinsic_reward = eta * 0.5 * ((pred_next - actual_next) ** 2).sum(dim=-1)
        return intrinsic_reward


class ICMCallback(DefaultCallbacks):
    """Callback to add ICM intrinsic rewards during training."""

    def __init__(self, icm_model: ICMModel, intrinsic_weight: float = 0.01):
        super().__init__()
        self.icm_model = icm_model
        self.intrinsic_weight = intrinsic_weight

    def on_postprocess_trajectory(
        self,
        *,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs,
    ):
        # Compute intrinsic rewards
        obs = torch.tensor(postprocessed_batch["obs"])
        next_obs = torch.tensor(postprocessed_batch["new_obs"])
        actions = torch.tensor(postprocessed_batch["actions"])

        intrinsic = self.icm_model.compute_intrinsic_reward(
            obs, next_obs, actions, eta=self.intrinsic_weight
        )

        # Add to extrinsic rewards
        postprocessed_batch["rewards"] += intrinsic.numpy()
```

### Random Network Distillation (RND)

```python
class RNDModel(nn.Module):
    """
    Random Network Distillation

    Uses prediction error of a random target network as intrinsic reward.
    Simpler than ICM, effective for high-dimensional observations.
    """

    def __init__(self, obs_dim: int, feature_dim: int = 512):
        super().__init__()

        # Target network (random, fixed)
        self.target = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

        # Predictor network (learned)
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute intrinsic reward from prediction error."""
        with torch.no_grad():
            target_features = self.target(obs)

        predicted_features = self.predictor(obs)

        # MSE between predictor and target
        error = ((predicted_features - target_features) ** 2).sum(dim=-1)

        if normalize:
            # Running normalization (simplified)
            error = error / (error.std() + 1e-8)

        return error
```

## Custom RLModule Implementation Patterns

### Graph Neural Network RLModule

For environments with graph-structured observations (molecules, networks, multi-agent):

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.columns import Columns

class GraphRLModule(TorchRLModule):
    """
    RLModule with Graph Neural Network encoder.

    Expects observations as graph data with node features and edge indices.
    """

    def setup(self):
        node_dim = self.observation_space["node_features"].shape[-1]
        hidden_dim = 128
        action_dim = self.action_space.n

        # Graph convolution layers
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Policy and value heads
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _forward_train(self, batch, **kwargs):
        # Extract graph components from batch
        x = batch[Columns.OBS]["node_features"]  # (total_nodes, node_dim)
        edge_index = batch[Columns.OBS]["edge_index"]  # (2, num_edges)
        batch_idx = batch[Columns.OBS]["batch"]  # (total_nodes,)

        # Graph convolutions with ReLU
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))

        # Global pooling to get graph-level representation
        graph_features = global_mean_pool(x, batch_idx)

        # Policy and value outputs
        action_logits = self.policy_head(graph_features)
        values = self.value_head(graph_features)

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.VF_PREDS: values.squeeze(-1),
        }

    def _forward_inference(self, batch, **kwargs):
        output = self._forward_train(batch, **kwargs)
        return {Columns.ACTION_DIST_INPUTS: output[Columns.ACTION_DIST_INPUTS]}

    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)
```

### Transformer-Based RLModule

For sequential decision-making with long-range dependencies:

```python
class TransformerRLModule(TorchRLModule):
    """
    RLModule with Transformer encoder for sequence modeling.

    Suitable for partially observable environments or decision transformers.
    """

    def setup(self):
        obs_dim = self.observation_space.shape[-1]
        action_dim = self.action_space.n

        self.embed_dim = 128
        self.num_heads = 4
        self.num_layers = 3
        self.max_seq_len = 100

        # Input embedding
        self.obs_embedding = nn.Linear(obs_dim, self.embed_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.max_seq_len, self.embed_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Output heads
        self.policy_head = nn.Linear(self.embed_dim, action_dim)
        self.value_head = nn.Linear(self.embed_dim, 1)

    def _forward_train(self, batch, **kwargs):
        obs = batch[Columns.OBS]  # (batch, seq_len, obs_dim)
        batch_size, seq_len, _ = obs.shape

        # Embed observations
        embedded = self.obs_embedding(obs)

        # Add positional encoding
        embedded = embedded + self.pos_encoding[:, :seq_len, :]

        # Create causal mask for autoregressive processing
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

        # Transformer forward
        encoded = self.transformer(embedded, mask=causal_mask)

        # Use last position for policy/value (or all positions for training)
        return {
            Columns.ACTION_DIST_INPUTS: self.policy_head(encoded),
            Columns.VF_PREDS: self.value_head(encoded).squeeze(-1),
        }
```

### Multi-Modal RLModule

For environments with multiple observation modalities (images + proprioception):

```python
class MultiModalRLModule(TorchRLModule):
    """
    RLModule that fuses multiple observation modalities.

    Handles image observations and vector observations separately,
    then fuses representations.
    """

    def setup(self):
        # Image encoder (CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )

        # Vector encoder (MLP)
        vector_dim = self.observation_space["vector"].shape[0]
        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
        )

        # Output heads
        action_dim = self.action_space.shape[0]  # Continuous actions
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)

    def _forward_train(self, batch, **kwargs):
        # Extract modalities
        images = batch[Columns.OBS]["image"]  # (batch, 3, 84, 84)
        vectors = batch[Columns.OBS]["vector"]  # (batch, vector_dim)

        # Encode each modality
        image_features = self.image_encoder(images)
        vector_features = self.vector_encoder(vectors)

        # Fuse representations
        fused = self.fusion(torch.cat([image_features, vector_features], dim=-1))

        # Outputs for continuous action distribution
        mean = self.mean_head(fused)
        log_std = self.log_std_head(fused).clamp(-20, 2)

        return {
            Columns.ACTION_DIST_INPUTS: {"mean": mean, "log_std": log_std},
            Columns.VF_PREDS: self.value_head(fused).squeeze(-1),
        }
```

## Reward Model Integration for RLHF

Integrate learned reward models for reinforcement learning from human feedback:

```python
import torch
import torch.nn as nn
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class RewardModel(nn.Module):
    """Learned reward model from human preferences."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict reward from observation-action pair."""
        inputs = torch.cat([obs, action], dim=-1)
        return self.encoder(inputs).squeeze(-1)


class RewardModelCallback(DefaultCallbacks):
    """Replace environment rewards with learned reward model predictions."""

    def __init__(self, reward_model_path: str, blend_ratio: float = 1.0):
        super().__init__()
        self.reward_model = torch.load(reward_model_path)
        self.reward_model.eval()
        self.blend_ratio = blend_ratio  # 1.0 = full learned reward

    def on_postprocess_trajectory(
        self,
        *,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        **kwargs,
    ):
        with torch.no_grad():
            obs = torch.tensor(postprocessed_batch["obs"], dtype=torch.float32)
            actions = torch.tensor(postprocessed_batch["actions"], dtype=torch.float32)

            learned_rewards = self.reward_model(obs, actions).numpy()

        # Blend learned and environment rewards
        env_rewards = postprocessed_batch["rewards"]
        postprocessed_batch["rewards"] = (
            self.blend_ratio * learned_rewards +
            (1 - self.blend_ratio) * env_rewards
        )
```

## Model-Based RL Approaches

### World Model Integration

Integrate learned dynamics models for planning:

```python
class WorldModel(nn.Module):
    """Learned dynamics model for model-based RL."""

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 64):
        super().__init__()

        # Encoder: obs -> latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # Dynamics: latent + action -> next_latent
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # Decoder: latent -> obs
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim),
        )

        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def predict(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next latent, observation, and reward."""
        inputs = torch.cat([latent, action], dim=-1)
        next_latent = self.dynamics(inputs)
        next_obs = self.decoder(next_latent)
        reward = self.reward_predictor(inputs).squeeze(-1)
        return next_latent, next_obs, reward

    def rollout(
        self,
        initial_obs: torch.Tensor,
        actions: torch.Tensor,  # (batch, horizon, action_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rollout dynamics model for planning."""
        batch_size, horizon, _ = actions.shape

        latent = self.encode(initial_obs)
        predicted_rewards = []
        predicted_obs = []

        for t in range(horizon):
            latent, obs, reward = self.predict(latent, actions[:, t])
            predicted_rewards.append(reward)
            predicted_obs.append(obs)

        return (
            torch.stack(predicted_obs, dim=1),
            torch.stack(predicted_rewards, dim=1),
        )
```

### Dyna-Style Training

Augment real experience with simulated experience:

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class DynaCallback(DefaultCallbacks):
    """Generate synthetic experience using world model."""

    def __init__(self, world_model: WorldModel, synthetic_ratio: float = 0.5):
        super().__init__()
        self.world_model = world_model
        self.synthetic_ratio = synthetic_ratio

    def on_learn_on_batch(
        self,
        *,
        policy,
        train_batch,
        result,
        **kwargs,
    ):
        # Generate synthetic batch
        batch_size = len(train_batch["obs"])
        synthetic_size = int(batch_size * self.synthetic_ratio)

        if synthetic_size > 0:
            # Sample starting states from real experience
            start_indices = torch.randint(0, batch_size, (synthetic_size,))
            start_obs = torch.tensor(train_batch["obs"][start_indices])

            # Generate synthetic rollouts
            with torch.no_grad():
                # Sample actions from current policy
                action_dist = policy.compute_actions(start_obs)
                actions = action_dist.sample()

                # Predict transitions
                latent = self.world_model.encode(start_obs)
                _, next_obs, rewards = self.world_model.predict(latent, actions)

            # Add synthetic experience to batch (simplified)
            result["synthetic_samples"] = synthetic_size
```

This reference guide provides comprehensive implementation details for advanced RL techniques in RLlib. Consult the main SKILL.md for workflow guidance and common pitfalls.
