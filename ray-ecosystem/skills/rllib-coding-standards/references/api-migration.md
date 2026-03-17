# RLlib API Migration Guide

This reference provides comprehensive migration guidance from the old RLlib API stack to the new API stack (Ray 2.53+). Use this document when migrating existing RLlib code or understanding the differences between API versions.

## Overview

Ray 2.53 introduced the new API stack as the default for RLlib. The new stack provides:

- **AlgorithmConfig builders** instead of dictionary-based configuration
- **RLModule** instead of ModelV2 for neural network definitions
- **EnvRunner** instead of RolloutWorker for environment interaction
- **Learner** for centralized training logic
- **PyTorch-only** support (TensorFlow deprecated)

## Complete Old to New API Mapping

### Algorithm Classes

| Old API | New API | Notes |
|---------|---------|-------|
| `PPOTrainer` | `PPOConfig().build()` | Trainer classes removed |
| `DQNTrainer` | `DQNConfig().build()` | Use config builder pattern |
| `SACTrainer` | `SACConfig().build()` | All algorithms follow same pattern |
| `A3CTrainer` | `A3CConfig().build()` | |
| `IMPALATrainer` | `IMPALAConfig().build()` | |
| `APPOTrainer` | `APPOConfig().build()` | |

### Configuration Parameters

| Old Parameter | New Parameter | Location |
|---------------|---------------|----------|
| `num_workers` | `num_env_runners` | `.env_runners()` |
| `num_rollout_workers` | `num_env_runners` | `.env_runners()` |
| `num_cpus_per_worker` | `num_cpus_per_env_runner` | `.env_runners()` |
| `num_gpus_per_worker` | `num_gpus_per_env_runner` | `.env_runners()` |
| `train_batch_size` | `train_batch_size_per_learner` | `.training()` |
| `rollout_fragment_length` | `rollout_fragment_length` | `.env_runners()` |
| `batch_mode` | `batch_mode` | `.env_runners()` |
| `framework` | `framework` | `.framework()` (torch only) |

### Model Classes

| Old API | New API | Notes |
|---------|---------|-------|
| `ModelV2` | `RLModule` | Base class for all models |
| `TorchModelV2` | `TorchRLModule` | PyTorch-specific implementation |
| `TFModelV2` | N/A | TensorFlow not supported |
| `RecurrentNetwork` | `TorchRLModule` with RNN | Implement RNN in setup() |
| `ActionDistribution` | Built-in distributions | Use Columns.ACTION_DIST_INPUTS |

### Worker Classes

| Old API | New API | Notes |
|---------|---------|-------|
| `RolloutWorker` | `EnvRunner` | Handles env stepping |
| `WorkerSet` | `EnvRunnerGroup` | Manages EnvRunner actors |
| `LocalWorker` | Local `EnvRunner` | `num_env_runners=0` |

## num_workers to num_env_runners Migration

### Before (Old API)

```python
from ray.rllib.agents.ppo import PPOTrainer

config = {
    "env": "CartPole-v1",
    "num_workers": 4,
    "num_cpus_per_worker": 1,
    "num_gpus_per_worker": 0,
    "num_envs_per_worker": 1,
    "rollout_fragment_length": 200,
    "train_batch_size": 4000,
    "framework": "torch",
}

trainer = PPOTrainer(config=config)
result = trainer.train()
trainer.stop()
```

### After (New API)

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .framework("torch")
    .env_runners(
        num_env_runners=4,
        num_cpus_per_env_runner=1,
        num_gpus_per_env_runner=0,
        num_envs_per_env_runner=1,
        rollout_fragment_length=200,
    )
    .training(
        train_batch_size_per_learner=4000,
    )
)

algo = config.build()
result = algo.train()
algo.stop()
```

### Key Differences

1. **Terminology**: `worker` becomes `env_runner` throughout the codebase
2. **Configuration method**: Dictionary replaced with fluent builder methods
3. **Batch size semantics**: `train_batch_size` was total across all workers; `train_batch_size_per_learner` is per-learner
4. **Algorithm instantiation**: `PPOTrainer(config)` becomes `PPOConfig().build()`

## ModelV2 to RLModule Migration

### Before (Old API - ModelV2)

```python
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn

class CustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        hidden_dim = model_config.get("custom_model_config", {}).get("hidden_dim", 64)
        input_dim = obs_space.shape[0]

        self._hidden_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self._policy_head = nn.Linear(hidden_dim, num_outputs)
        self._value_head = nn.Linear(hidden_dim, 1)
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        self._features = self._hidden_layers(obs)
        logits = self._policy_head(self._features)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_head(self._features).squeeze(-1)

# Registration
ModelCatalog.register_custom_model("my_model", CustomModel)

# Usage in config
config = {
    "model": {
        "custom_model": "my_model",
        "custom_model_config": {"hidden_dim": 128},
    },
}
```

### After (New API - RLModule)

```python
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
import torch
import torch.nn as nn
from typing import Any

class CustomRLModule(TorchRLModule):
    def setup(self):
        """Initialize neural network components.

        Available attributes:
        - self.observation_space: The observation space
        - self.action_space: The action space
        - self.inference_only: Whether this module is inference-only
        - self.model_config: Configuration dictionary
        """
        input_dim = self.observation_space.shape[0]
        hidden_dim = self.model_config.get("hidden_dim", 64)

        # For discrete actions
        if hasattr(self.action_space, "n"):
            output_dim = self.action_space.n
        else:
            output_dim = self.action_space.shape[0] * 2  # Mean and log_std

        self._encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self._policy_head = nn.Linear(hidden_dim, output_dim)
        self._value_head = nn.Linear(hidden_dim, 1)

    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Shared forward pass for both inference and exploration."""
        obs = batch[Columns.OBS].float()
        features = self._encoder(obs)
        action_logits = self._policy_head(features)
        vf_preds = self._value_head(features).squeeze(-1)

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.VF_PREDS: vf_preds,
        }

    def _forward_inference(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Forward pass for inference (deployment)."""
        return self._forward(batch, **kwargs)

    def _forward_exploration(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Forward pass for exploration (training data collection)."""
        return self._forward(batch, **kwargs)


# Usage in config (New API)
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .framework("torch")
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[128, 128],
            fcnet_activation="relu",
        ),
        # Or use custom module:
        # rl_module_spec=RLModuleSpec(module_class=CustomRLModule),
    )
)
```

### Key Migration Steps

1. **Inherit from `TorchRLModule`** instead of `TorchModelV2` and `nn.Module`
2. **Move initialization to `setup()`** instead of `__init__`
3. **Replace `forward()` with `_forward()`**, `_forward_inference()`, and `_forward_exploration()`
4. **Use `Columns` constants** for batch keys (`Columns.OBS`, `Columns.ACTION_DIST_INPUTS`, etc.)
5. **Remove `value_function()` method** - return `Columns.VF_PREDS` from `_forward()` instead
6. **No model registration needed** - use `RLModuleSpec` in config

## Dict Config to AlgorithmConfig Builder Migration

### Before (Old API - Dict Config)

```python
from ray.rllib.agents.ppo import PPOTrainer

config = {
    "env": "CartPole-v1",
    "framework": "torch",

    # Training
    "gamma": 0.99,
    "lr": 0.0003,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 10,
    "lambda": 0.95,
    "clip_param": 0.2,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,

    # Workers
    "num_workers": 4,
    "num_cpus_per_worker": 1,
    "num_envs_per_worker": 1,
    "rollout_fragment_length": 200,

    # Resources
    "num_gpus": 1,

    # Model
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },

    # Evaluation
    "evaluation_interval": 10,
    "evaluation_num_workers": 1,
    "evaluation_config": {
        "explore": False,
    },

    # Callbacks
    "callbacks": MyCallbacks,
}

trainer = PPOTrainer(config=config)
```

### After (New API - AlgorithmConfig Builder)

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

config = (
    PPOConfig()
    # Environment
    .environment(
        env="CartPole-v1",
    )
    # Framework
    .framework("torch")
    # Training hyperparameters
    .training(
        gamma=0.99,
        lr=0.0003,
        train_batch_size_per_learner=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
        lambda_=0.95,  # Note: underscore suffix for Python keyword
        clip_param=0.2,
        vf_loss_coeff=0.5,
        entropy_coeff=0.01,
    )
    # Environment runners (formerly workers)
    .env_runners(
        num_env_runners=4,
        num_cpus_per_env_runner=1,
        num_envs_per_env_runner=1,
        rollout_fragment_length=200,
    )
    # Resources
    .resources(
        num_gpus=1,
    )
    # Model configuration
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[64, 64],
            fcnet_activation="relu",
        ),
    )
    # Evaluation
    .evaluation(
        evaluation_interval=10,
        evaluation_num_env_runners=1,
        evaluation_config=PPOConfig.overrides(explore=False),
    )
    # Callbacks
    .callbacks(MyCallbacks)
)

algo = config.build()
```

### Builder Method Reference

| Config Section | Builder Method | Common Parameters |
|----------------|----------------|-------------------|
| Environment | `.environment()` | `env`, `env_config`, `observation_space`, `action_space` |
| Framework | `.framework()` | `"torch"` (only supported value) |
| Training | `.training()` | `gamma`, `lr`, `train_batch_size_per_learner`, algorithm-specific |
| EnvRunners | `.env_runners()` | `num_env_runners`, `rollout_fragment_length`, `batch_mode` |
| Learners | `.learners()` | `num_learners`, `num_gpus_per_learner` |
| Resources | `.resources()` | `num_gpus`, `num_cpus` |
| Model | `.rl_module()` | `model_config`, `rl_module_spec` |
| Evaluation | `.evaluation()` | `evaluation_interval`, `evaluation_num_env_runners` |
| Callbacks | `.callbacks()` | Callback class or callable |
| Multi-Agent | `.multi_agent()` | `policies`, `policy_mapping_fn` |
| Offline | `.offline_data()` | `input_`, `actions_in_input_normalized` |
| Exploration | `.exploration()` | `explore`, `exploration_config` |

## TensorFlow to PyTorch Migration

### Key Changes

1. **Framework setting**: Change `framework="tf"` or `framework="tf2"` to `framework="torch"`
2. **Model classes**: Replace `TFModelV2` with `TorchRLModule`
3. **Tensor operations**: Replace TensorFlow ops with PyTorch equivalents
4. **Eager execution**: PyTorch is eager by default (no session management)

### TensorFlow Operations to PyTorch

| TensorFlow | PyTorch |
|------------|---------|
| `tf.constant()` | `torch.tensor()` |
| `tf.Variable()` | `torch.nn.Parameter()` |
| `tf.keras.layers.Dense()` | `torch.nn.Linear()` |
| `tf.nn.relu()` | `torch.nn.ReLU()` or `torch.relu()` |
| `tf.reduce_mean()` | `torch.mean()` |
| `tf.concat()` | `torch.cat()` |
| `tf.reshape()` | `tensor.reshape()` or `torch.reshape()` |
| `tf.squeeze()` | `tensor.squeeze()` or `torch.squeeze()` |
| `@tf.function` | Not needed (eager by default) |

### Before (TensorFlow)

```python
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

class TFCustomModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.policy = tf.keras.layers.Dense(num_outputs)
        self.value = tf.keras.layers.Dense(1)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        x = self.dense1(x)
        x = self.dense2(x)
        self._features = x
        return self.policy(x), state

    def value_function(self):
        return tf.squeeze(self.value(self._features), -1)
```

### After (PyTorch)

```python
import torch
import torch.nn as nn
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns

class TorchCustomModule(TorchRLModule):
    def setup(self):
        input_dim = self.observation_space.shape[0]
        output_dim = self.action_space.n

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, output_dim)
        self.value_head = nn.Linear(64, 1)

    def _forward(self, batch, **kwargs):
        x = batch[Columns.OBS].float()
        features = self.encoder(x)
        return {
            Columns.ACTION_DIST_INPUTS: self.policy_head(features),
            Columns.VF_PREDS: self.value_head(features).squeeze(-1),
        }
```

## Version Compatibility Matrix

| Ray Version | Old API Stack | New API Stack | Default | Notes |
|-------------|---------------|---------------|---------|-------|
| < 2.0 | Supported | N/A | Old | Pre-RLlib rewrite |
| 2.0 - 2.5 | Supported | Experimental | Old | New API introduced |
| 2.6 - 2.9 | Supported | Beta | Old | New API maturing |
| 2.10 - 2.52 | Supported | Stable | Old | Both stacks work |
| 2.53+ | Deprecated | Stable | **New** | New API is default |
| 3.0+ (planned) | Removed | Required | New | Old API removed |

### Enabling/Disabling API Stacks

```python
from ray.rllib.algorithms.ppo import PPOConfig

# New API stack (default in 2.53+)
config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
)

# Old API stack (for migration period only)
config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
)
```

## Common Migration Errors and Fixes

### Error: `AttributeError: 'PPOConfig' has no attribute 'num_workers'`

**Cause**: Using old parameter name directly on config object.

**Fix**: Use the `.env_runners()` method with `num_env_runners`.

```python
# Wrong
config.num_workers = 4

# Correct
config = PPOConfig().env_runners(num_env_runners=4)
```

### Error: `ValueError: Unknown framework: tf`

**Cause**: TensorFlow is not supported in new API stack.

**Fix**: Use PyTorch framework.

```python
# Wrong
config = PPOConfig().framework("tf")

# Correct
config = PPOConfig().framework("torch")
```

### Error: `ImportError: cannot import name 'PPOTrainer'`

**Cause**: Trainer classes have been removed.

**Fix**: Use `PPOConfig().build()` pattern.

```python
# Wrong
from ray.rllib.agents.ppo import PPOTrainer
trainer = PPOTrainer(config=config_dict)

# Correct
from ray.rllib.algorithms.ppo import PPOConfig
algo = PPOConfig().environment("CartPole-v1").build()
```

### Error: `TypeError: forward() got unexpected keyword argument 'input_dict'`

**Cause**: Using ModelV2 signature in RLModule.

**Fix**: Update to RLModule method signatures.

```python
# Wrong (ModelV2 style)
def forward(self, input_dict, state, seq_lens):
    obs = input_dict["obs"]
    ...

# Correct (RLModule style)
def _forward(self, batch, **kwargs):
    obs = batch[Columns.OBS]
    ...
```

### Error: `KeyError: 'train_batch_size'`

**Cause**: Old batch size parameter name.

**Fix**: Use `train_batch_size_per_learner` in `.training()`.

```python
# Wrong
config = {"train_batch_size": 4000}

# Correct
config = PPOConfig().training(train_batch_size_per_learner=4000)
```

### Error: `AttributeError: 'Algorithm' has no attribute 'workers'`

**Cause**: `workers` attribute renamed to `env_runner_group`.

**Fix**: Use new attribute name.

```python
# Wrong
workers = algo.workers

# Correct
env_runner_group = algo.env_runner_group
```

### Error: Environment returns 4-tuple instead of 5-tuple

**Cause**: Using old Gym API instead of Gymnasium.

**Fix**: Update environment to Gymnasium 5-tuple protocol.

```python
# Wrong (old Gym)
def step(self, action):
    obs = ...
    reward = ...
    done = ...
    info = {}
    return obs, reward, done, info  # 4-tuple

# Correct (Gymnasium)
def step(self, action):
    obs = ...
    reward = ...
    terminated = ...  # Episode ended naturally
    truncated = ...   # Episode ended due to time limit
    info = {}
    return obs, reward, terminated, truncated, info  # 5-tuple
```

### Warning: `DeprecationWarning: ModelV2 is deprecated`

**Cause**: Using legacy model API.

**Fix**: Migrate to RLModule. See the ModelV2 to RLModule migration section above.

## Migration Checklist

Use this checklist when migrating existing RLlib code:

```
[ ] Replace Trainer classes with Config().build() pattern
[ ] Change framework from "tf"/"tf2" to "torch"
[ ] Update num_workers to num_env_runners
[ ] Update train_batch_size to train_batch_size_per_learner
[ ] Migrate ModelV2 to TorchRLModule
[ ] Update environment to Gymnasium 5-tuple protocol
[ ] Replace dict config with AlgorithmConfig builder
[ ] Update evaluation_num_workers to evaluation_num_env_runners
[ ] Replace workers attribute access with env_runner_group
[ ] Test with api_stack() explicitly enabled
[ ] Run smoke tests to verify functionality
[ ] Update import paths (ray.rllib.agents -> ray.rllib.algorithms)
```
