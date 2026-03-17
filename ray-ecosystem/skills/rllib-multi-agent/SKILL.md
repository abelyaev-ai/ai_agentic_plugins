---
name: rllib-multi-agent
description: >
  This skill should be used when the user asks to "set up multi-agent RL",
  "configure policy mapping", "coordinate multiple agents",
  "shape rewards for MARL", or works with RLlib multi-agent.
---

## Purpose

Provide patterns and guidance for implementing multi-agent reinforcement learning (MARL) systems using RLlib's new API stack (2.53+). Cover environment design with per-agent observation and action spaces, policy mapping strategies, shared versus independent policies, agent grouping, and multi-agent training and evaluation workflows.

## Prerequisites

Before implementing multi-agent systems, resolve the RLlib multi-agent documentation via Context7:

1. Call `resolve-library-id` with `libraryName: "ray rllib"` and `query: "multi-agent reinforcement learning policy mapping MultiAgentEnv"`.
2. Call `query-docs` with the resolved library ID and queries for:
   - `MultiAgentEnv observation_space action_space per agent`
   - `multi_agent policy_mapping_fn policies configuration`
   - `agent grouping hierarchical Q-Mix`

Ground all implementations in the latest API surface rather than memorized patterns.

## Core Workflow

### Step 1: Design the MultiAgentEnv

Create a custom environment by subclassing `ray.rllib.env.multi_agent_env.MultiAgentEnv`. Define per-agent observation and action spaces explicitly.

**Key attributes to implement:**

- `agents` / `possible_agents`: List of agent IDs currently active and all possible agent IDs that may appear.
- `observation_spaces`: Dictionary mapping `AgentID` to `gymnasium.Space` objects.
- `action_spaces`: Dictionary mapping `AgentID` to `gymnasium.Space` objects.

```python
import gymnasium as gym
import numpy as np
from typing import Any, Optional
from ray.rllib.env.multi_agent_env import MultiAgentEnv

AgentID = str

class CooperativeNavigation(MultiAgentEnv):
    """Multi-agent environment where agents navigate to targets."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__()
        config = config or {}
        self.num_agents = config.get("num_agents", 3)
        self.grid_size = config.get("grid_size", 10)

        # Define all possible agents (constant across episodes)
        self.possible_agents = [f"agent_{i}" for i in range(self.num_agents)]
        # Active agents (can change during episode)
        self.agents = list(self.possible_agents)

        # Per-agent observation spaces
        # Each agent observes: own position (2), target position (2), other agents (2 * (n-1))
        obs_dim = 2 + 2 + 2 * (self.num_agents - 1)
        self.observation_spaces = {
            agent_id: gym.spaces.Box(
                low=0.0, high=float(self.grid_size), shape=(obs_dim,), dtype=np.float32
            )
            for agent_id in self.possible_agents
        }

        # Per-agent action spaces (4 discrete movement directions)
        self.action_spaces = {
            agent_id: gym.spaces.Discrete(4)
            for agent_id in self.possible_agents
        }

        self._positions: dict[str, np.ndarray] = {}
        self._targets: dict[str, np.ndarray] = {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[AgentID, np.ndarray], dict[AgentID, dict[str, Any]]]:
        super().reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)

        # Initialize positions and targets
        rng = np.random.default_rng(seed)
        for agent_id in self.agents:
            self._positions[agent_id] = rng.uniform(0, self.grid_size, size=2).astype(np.float32)
            self._targets[agent_id] = rng.uniform(0, self.grid_size, size=2).astype(np.float32)

        obs = self._get_observations()
        infos = {agent_id: {} for agent_id in self.agents}
        return obs, infos

    def step(
        self,
        action_dict: dict[AgentID, int],
    ) -> tuple[
        dict[AgentID, np.ndarray],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        """Execute actions and return Gymnasium 5-tuple per agent."""
        rewards: dict[AgentID, float] = {}
        terminateds: dict[AgentID, bool] = {}
        truncateds: dict[AgentID, bool] = {}
        infos: dict[AgentID, dict[str, Any]] = {}

        # Process actions for each active agent
        for agent_id, action in action_dict.items():
            self._apply_action(agent_id, action)
            rewards[agent_id] = self._compute_reward(agent_id)
            terminateds[agent_id] = self._check_done(agent_id)
            truncateds[agent_id] = False
            infos[agent_id] = {}

        # Special keys for environment-level done signals
        terminateds["__all__"] = all(terminateds.get(a, False) for a in self.agents)
        truncateds["__all__"] = False

        obs = self._get_observations()
        return obs, rewards, terminateds, truncateds, infos

    def _get_observations(self) -> dict[AgentID, np.ndarray]:
        """Build per-agent observations including other agents' positions."""
        obs = {}
        for agent_id in self.agents:
            own_pos = self._positions[agent_id]
            target_pos = self._targets[agent_id]
            other_positions = np.concatenate([
                self._positions[other_id]
                for other_id in self.agents if other_id != agent_id
            ])
            obs[agent_id] = np.concatenate([own_pos, target_pos, other_positions]).astype(np.float32)
        return obs

    def _apply_action(self, agent_id: AgentID, action: int) -> None:
        """Apply movement action to agent."""
        moves = {0: [0, 1], 1: [0, -1], 2: [1, 0], 3: [-1, 0]}
        delta = np.array(moves[action], dtype=np.float32)
        self._positions[agent_id] = np.clip(
            self._positions[agent_id] + delta, 0, self.grid_size
        )

    def _compute_reward(self, agent_id: AgentID) -> float:
        """Negative distance to target."""
        distance = np.linalg.norm(self._positions[agent_id] - self._targets[agent_id])
        return float(-distance)

    def _check_done(self, agent_id: AgentID) -> bool:
        """Agent is done when it reaches its target."""
        distance = np.linalg.norm(self._positions[agent_id] - self._targets[agent_id])
        return bool(distance < 0.5)
```

### Step 2: Register the Environment

Register the custom environment with Ray Tune before configuring the algorithm.

```python
from ray import tune

tune.register_env(
    "cooperative_navigation",
    lambda config: CooperativeNavigation(config),
)
```

### Step 3: Define the Policy Mapping Function

The policy mapping function determines which policy controls each agent. It receives the agent ID, the current episode, and optional keyword arguments.

**Common patterns:**

1. **One policy per agent** (independent learning):
   ```python
   policy_mapping_fn = lambda agent_id, episode, **kwargs: agent_id
   ```

2. **Shared policy** (parameter sharing):
   ```python
   policy_mapping_fn = lambda agent_id, episode, **kwargs: "shared_policy"
   ```

3. **Role-based mapping** (heterogeneous agents):
   ```python
   def policy_mapping_fn(agent_id: str, episode, **kwargs) -> str:
       if agent_id.startswith("predator"):
           return "predator_policy"
       elif agent_id.startswith("prey"):
           return "prey_policy"
       return "default_policy"
   ```

4. **Dynamic mapping** (conditional on episode state):
   ```python
   import random

   def policy_mapping_fn(agent_id: str, episode, **kwargs) -> str:
       # Randomly assign to one of several policies for diversity
       return random.choice(["policy_v1", "policy_v2", "policy_v3"])
   ```

### Step 4: Configure Multi-Agent Training

Use `AlgorithmConfig.multi_agent()` to define policies and the mapping function. Combine with `rl_module()` to specify per-policy neural network architectures.

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.policy.policy import PolicySpec

config = (
    PPOConfig()
    .environment(
        env="cooperative_navigation",
        env_config={"num_agents": 3, "grid_size": 10},
    )
    .framework("torch")
    .env_runners(num_env_runners=4)
    .multi_agent(
        policies={
            "shared_policy": PolicySpec(),  # Single shared policy
        },
        policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        policies_to_train=["shared_policy"],  # Explicitly train this policy
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "shared_policy": RLModuleSpec(),
            }
        ),
    )
    .training(
        train_batch_size_per_learner=4000,
        lr=3e-4,
        gamma=0.99,
    )
)

algo = config.build()
```

### Step 5: Configure Independent Policies with Overrides

For heterogeneous agents requiring different hyperparameters or architectures, define multiple policies with per-module overrides.

```python
config = (
    PPOConfig()
    .environment(env="predator_prey_env")
    .framework("torch")
    .multi_agent(
        policies={"predator", "prey"},  # Can use set of IDs
        policy_mapping_fn=lambda aid, episode, **kw: (
            "predator" if aid.startswith("predator") else "prey"
        ),
        algorithm_config_overrides_per_module={
            "predator": PPOConfig.overrides(gamma=0.95, lr=1e-4),
            "prey": PPOConfig.overrides(gamma=0.99, lr=5e-4),
        },
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "predator": RLModuleSpec(model_config={"fcnet_hiddens": [256, 256]}),
                "prey": RLModuleSpec(model_config={"fcnet_hiddens": [128, 128]}),
            }
        ),
    )
)
```

### Step 6: Agent Grouping for Algorithms like Q-Mix

Group agents into logical units for algorithms that require joint action spaces (e.g., QMIX, VDN). Use `MultiAgentEnv.with_agent_groups()`.

```python
from ray.rllib.env.wrappers.group_agents_wrapper import GroupAgentsWrapper

# Original environment with individual agents
base_env = CooperativeNavigation({"num_agents": 6})

# Group into two teams of 3
grouped_env = base_env.with_agent_groups(
    groups={
        "team_red": ["agent_0", "agent_1", "agent_2"],
        "team_blue": ["agent_3", "agent_4", "agent_5"],
    }
)
# grouped_env now has agents: ["team_red", "team_blue"]
# Observations and actions are Tuples of the individual agents' spaces
# Rewards are summed within each group
```

### Step 7: Multi-Agent Training Loop

Execute training iterations and monitor per-policy metrics.

```python
import ray

ray.init()

try:
    algo = config.build()

    for iteration in range(100):
        result = algo.train()

        # Extract per-policy metrics
        policy_reward_mean = result.get("env_runners", {}).get("policy_reward_mean", {})

        print(f"Iteration {iteration}")
        for policy_id, reward in policy_reward_mean.items():
            print(f"  {policy_id}: mean_reward={reward:.2f}")

        # Save checkpoint every 10 iterations
        if iteration % 10 == 0:
            checkpoint_path = algo.save()
            print(f"  Checkpoint saved: {checkpoint_path}")

finally:
    algo.stop()
    ray.shutdown()
```

### Step 8: Multi-Agent Evaluation

Evaluate trained policies with deterministic actions or against specific opponent policies.

```python
from ray.rllib.algorithms.algorithm import Algorithm

# Load from checkpoint
algo = Algorithm.from_checkpoint("/path/to/checkpoint")

# Run evaluation episodes
eval_config = algo.config.copy(deep=True)
eval_config.env_runners(num_env_runners=1)
eval_config.evaluation(
    evaluation_interval=1,
    evaluation_num_env_runners=2,
    evaluation_duration=10,
    evaluation_duration_unit="episodes",
)

eval_algo = eval_config.build()
eval_results = eval_algo.evaluate()

print("Evaluation results:")
for policy_id, metrics in eval_results.get("env_runners", {}).get("policy_reward_mean", {}).items():
    print(f"  {policy_id}: {metrics}")

eval_algo.stop()
```

## Common Pitfalls

- **Mismatched spaces**: Ensure `observation_spaces` and `action_spaces` dictionaries contain entries for all agents in `possible_agents`. RLlib validates these at environment creation.

- **Missing `__all__` keys**: The `terminateds` and `truncateds` dictionaries returned by `step()` must include `"__all__"` keys indicating whether the entire episode is done. Omitting these causes runtime errors.

- **Stale policy mapping**: The `policy_mapping_fn` is called at the start of each episode for each agent. If agents are dynamically added mid-episode, ensure the function handles unknown agent IDs gracefully.

- **Unregistered environments**: Always call `tune.register_env()` before building the algorithm. Passing the class directly works for single-agent but can cause serialization issues in multi-agent distributed settings.

- **Forgetting `policies_to_train`**: When some policies should remain frozen (e.g., opponent policies in self-play), explicitly set `policies_to_train` to avoid training all policies by default.

## Additional Resources

For advanced coordination strategies including centralized training with decentralized execution (CTDE), parameter sharing, self-play, and league training patterns, see:

- `references/coordination-patterns.md` in this skill directory
