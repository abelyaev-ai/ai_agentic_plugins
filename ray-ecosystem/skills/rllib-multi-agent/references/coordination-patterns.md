# Multi-Agent Coordination Patterns

This reference covers advanced coordination strategies for multi-agent reinforcement learning (MARL) systems in RLlib. These patterns address how agents share information, learn from each other, and coordinate their behaviors in cooperative, competitive, or mixed-motive settings.

## Centralized Training Decentralized Execution (CTDE)

CTDE is the dominant paradigm in modern MARL. During training, a centralized component has access to global state and all agents' observations. During execution, each agent acts using only its local observations.

### Why CTDE Works

- **Training stability**: Centralized critics or value functions can condition on global state, reducing variance in policy gradient estimates and addressing non-stationarity caused by other learning agents.
- **Deployment flexibility**: Decentralized execution requires no communication infrastructure at inference time, making deployment practical in bandwidth-constrained or latency-sensitive environments.
- **Scalability**: Agents can share a centralized critic during training but maintain independent policies, allowing the system to scale to many agents.

### Implementing CTDE in RLlib

RLlib supports CTDE through custom observation functions and centralized critic architectures.

**Approach 1: Observation Function for Centralized Information**

Use the `observation_fn` parameter to inject global state into training observations while keeping execution observations local.

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.sample_batch import SampleBatch

def centralized_observation_fn(
    agent_obs: dict,
    agent_id: str,
    episode,
    **kwargs,
) -> dict:
    """Augment local observation with global state during training."""
    # During training, include global state
    # During execution, this function is not called
    global_state = episode.get_state()  # Environment-specific
    return {
        "local_obs": agent_obs,
        "global_state": global_state,
    }

config = (
    PPOConfig()
    .environment(env="my_marl_env")
    .multi_agent(
        policies={"agent_policy"},
        policy_mapping_fn=lambda aid, ep, **kw: "agent_policy",
        observation_fn=centralized_observation_fn,
    )
)
```

**Approach 2: Custom RLModule with Centralized Critic**

Implement a custom `RLModule` that uses different observation spaces for the actor (local) and critic (global).

```python
import torch
from torch import nn
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.utils.annotations import override

class CentralizedCriticModule(TorchRLModule):
    """RLModule with decentralized actor and centralized critic."""

    def __init__(self, config):
        super().__init__(config)
        local_obs_dim = config.observation_space.shape[0]
        global_state_dim = config.model_config.get("global_state_dim", 64)
        action_dim = config.action_space.n

        # Decentralized actor (local observations only)
        self.actor = nn.Sequential(
            nn.Linear(local_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # Centralized critic (global state)
        self.critic = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    @override(TorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        """Inference uses only local observations for the actor."""
        local_obs = batch["obs"]["local_obs"]
        logits = self.actor(local_obs)
        return {"action_dist_inputs": logits}

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        """Training uses global state for the critic."""
        local_obs = batch["obs"]["local_obs"]
        global_state = batch["obs"]["global_state"]

        logits = self.actor(local_obs)
        value = self.critic(global_state)

        return {
            "action_dist_inputs": logits,
            "vf_preds": value.squeeze(-1),
        }
```

## Parameter Sharing Strategies

Parameter sharing reduces the number of learnable parameters and can accelerate learning by allowing agents to benefit from each other's experiences.

### Full Parameter Sharing

All agents use identical neural network weights. Differentiation comes from agent-specific observations (e.g., agent ID embeddings or local sensor readings).

```python
import numpy as np

def add_agent_id_to_obs(obs: np.ndarray, agent_id: str, num_agents: int) -> np.ndarray:
    """Concatenate one-hot agent ID to observation."""
    agent_idx = int(agent_id.split("_")[1])
    one_hot = np.zeros(num_agents, dtype=np.float32)
    one_hot[agent_idx] = 1.0
    return np.concatenate([obs, one_hot])

# Configure shared policy for all agents
config = (
    PPOConfig()
    .environment(
        env="team_env",
        env_config={"num_agents": 4},
    )
    .multi_agent(
        policies={"shared"},
        policy_mapping_fn=lambda aid, ep, **kw: "shared",
    )
)
```

**When to use full sharing:**
- Homogeneous agents with symmetric roles
- Limited training budget or sample efficiency concerns
- Emergent specialization is acceptable

### Partial Parameter Sharing

Agents share some layers (e.g., feature extractors) but have independent output heads.

```python
class PartiallySharedModule(TorchRLModule):
    """Shared encoder with agent-specific policy heads."""

    def __init__(self, config):
        super().__init__(config)
        obs_dim = config.observation_space.shape[0]
        action_dim = config.action_space.n
        num_agents = config.model_config.get("num_agents", 4)

        # Shared feature encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Agent-specific policy heads
        self.policy_heads = nn.ModuleDict({
            f"agent_{i}": nn.Linear(64, action_dim)
            for i in range(num_agents)
        })

        # Shared value head
        self.value_head = nn.Linear(64, 1)

    def forward(self, batch, agent_id: str):
        obs = batch["obs"]
        features = self.shared_encoder(obs)
        logits = self.policy_heads[agent_id](features)
        value = self.value_head(features)
        return logits, value
```

### No Parameter Sharing

Each agent has completely independent networks. Use when agents have fundamentally different roles, observation spaces, or action spaces.

```python
config = (
    PPOConfig()
    .multi_agent(
        policies={
            "goalkeeper": PolicySpec(
                observation_space=goalkeeper_obs_space,
                action_space=goalkeeper_action_space,
            ),
            "striker": PolicySpec(
                observation_space=striker_obs_space,
                action_space=striker_action_space,
            ),
            "midfielder": PolicySpec(
                observation_space=midfielder_obs_space,
                action_space=midfielder_action_space,
            ),
        },
        policy_mapping_fn=lambda aid, ep, **kw: aid.split("_")[0],
    )
)
```

## Communication Channels Between Agents

Explicit communication allows agents to share information beyond what is observable in the environment.

### Discrete Message Passing

Agents output discrete messages alongside actions. Messages from the previous timestep are included in observations.

```python
import gymnasium as gym
import numpy as np

class CommunicativeEnv(MultiAgentEnv):
    """Environment with discrete communication channels."""

    def __init__(self, config):
        super().__init__()
        self.num_agents = config.get("num_agents", 3)
        self.message_dim = config.get("message_dim", 8)
        base_obs_dim = config.get("obs_dim", 16)

        self.possible_agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.agents = list(self.possible_agents)

        # Observation includes own state + received messages
        obs_dim = base_obs_dim + self.message_dim * (self.num_agents - 1)
        self.observation_spaces = {
            aid: gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
            for aid in self.possible_agents
        }

        # Action space includes environment action + message
        self.action_spaces = {
            aid: gym.spaces.Dict({
                "action": gym.spaces.Discrete(4),
                "message": gym.spaces.MultiBinary(self.message_dim),
            })
            for aid in self.possible_agents
        }

        self._messages = {aid: np.zeros(self.message_dim) for aid in self.agents}

    def step(self, action_dict):
        # Extract environment actions and messages
        env_actions = {aid: act["action"] for aid, act in action_dict.items()}
        new_messages = {aid: act["message"] for aid, act in action_dict.items()}

        # Process environment step
        obs, rewards, terminateds, truncateds, infos = self._env_step(env_actions)

        # Update messages for next observation
        self._messages = new_messages

        # Include received messages in observations
        obs_with_messages = self._add_messages_to_obs(obs)

        return obs_with_messages, rewards, terminateds, truncateds, infos

    def _add_messages_to_obs(self, obs):
        """Concatenate messages from other agents to each observation."""
        obs_with_msg = {}
        for aid in self.agents:
            other_messages = np.concatenate([
                self._messages[other_aid]
                for other_aid in self.agents if other_aid != aid
            ])
            obs_with_msg[aid] = np.concatenate([obs[aid], other_messages])
        return obs_with_msg
```

### Learned Communication (CommNet, TarMAC)

Implement attention-based communication where agents learn what information to share.

```python
class AttentionCommunication(nn.Module):
    """Multi-head attention for inter-agent communication."""

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, agent_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_embeddings: (batch, num_agents, embed_dim)

        Returns:
            Contextualized embeddings: (batch, num_agents, embed_dim)
        """
        attended, _ = self.attention(
            agent_embeddings, agent_embeddings, agent_embeddings
        )
        return self.norm(agent_embeddings + attended)
```

## Reward Shaping for Cooperation vs Competition

The reward structure fundamentally shapes emergent agent behaviors.

### Cooperative Reward Structures

**Team reward (fully cooperative):**
All agents receive the same global reward signal.

```python
def compute_team_reward(agent_positions, goal_position):
    """All agents receive distance-based team reward."""
    min_distance = min(
        np.linalg.norm(pos - goal_position)
        for pos in agent_positions.values()
    )
    team_reward = -min_distance  # Negative distance to goal
    return {aid: team_reward for aid in agent_positions}
```

**Mixed individual and team rewards:**
Combine local and global objectives.

```python
def compute_mixed_reward(agent_id, local_reward, team_reward, alpha=0.5):
    """Weighted combination of individual and team rewards."""
    return alpha * local_reward + (1 - alpha) * team_reward
```

**Intrinsic social rewards:**
Encourage helpful behaviors through auxiliary rewards.

```python
def compute_social_reward(agent_id, helped_agents, help_bonus=0.1):
    """Bonus for actions that help teammates."""
    return help_bonus * len(helped_agents)
```

### Competitive Reward Structures

**Zero-sum rewards:**
One agent's gain is another's loss.

```python
def compute_zero_sum_rewards(winner_id, loser_id, agents):
    """Winner gets +1, loser gets -1."""
    rewards = {aid: 0.0 for aid in agents}
    rewards[winner_id] = 1.0
    rewards[loser_id] = -1.0
    return rewards
```

**Relative ranking rewards:**
Rewards based on performance relative to others.

```python
def compute_ranking_rewards(agent_scores):
    """Rewards proportional to rank among agents."""
    sorted_agents = sorted(agent_scores.items(), key=lambda x: -x[1])
    num_agents = len(sorted_agents)
    rewards = {}
    for rank, (aid, _) in enumerate(sorted_agents):
        # Linear scaling from 1.0 (best) to -1.0 (worst)
        rewards[aid] = 1.0 - 2.0 * rank / (num_agents - 1)
    return rewards
```

### Mixed-Motive Rewards

Agents have partially aligned incentives, requiring negotiation and strategic behavior.

```python
def compute_mixed_motive_rewards(agent_actions, resource_pool):
    """Social dilemma: cooperate for collective good or defect for individual gain."""
    cooperators = [aid for aid, act in agent_actions.items() if act == "cooperate"]
    defectors = [aid for aid, act in agent_actions.items() if act == "defect"]

    coop_bonus = resource_pool * 1.5 / len(agent_actions) if cooperators else 0
    defect_bonus = resource_pool * 0.5 / len(defectors) if defectors else 0

    rewards = {}
    for aid, action in agent_actions.items():
        if action == "cooperate":
            rewards[aid] = coop_bonus
        else:
            rewards[aid] = coop_bonus + defect_bonus  # Defectors exploit cooperators
    return rewards
```

## Self-Play Patterns

Self-play trains agents by having them compete against copies of themselves, enabling continuous improvement without external opponents.

### Basic Self-Play

The simplest form: current policy plays against itself.

```python
config = (
    PPOConfig()
    .environment(env="two_player_game")
    .multi_agent(
        policies={"main"},
        policy_mapping_fn=lambda aid, ep, **kw: "main",  # Both players use same policy
    )
)
```

### Frozen Self-Play

Periodically freeze a copy of the policy as a fixed opponent.

```python
from ray.rllib.policy.policy import PolicySpec

class SelfPlayCallback:
    """Callback to update frozen opponent policy."""

    def __init__(self, freeze_interval: int = 10):
        self.freeze_interval = freeze_interval
        self.iteration = 0

    def on_train_result(self, *, algorithm, result, **kwargs):
        self.iteration += 1
        if self.iteration % self.freeze_interval == 0:
            # Copy current weights to opponent policy
            main_weights = algorithm.get_policy("main").get_weights()
            algorithm.get_policy("opponent").set_weights(main_weights)

config = (
    PPOConfig()
    .environment(env="two_player_game")
    .multi_agent(
        policies={
            "main": PolicySpec(),
            "opponent": PolicySpec(),
        },
        policy_mapping_fn=lambda aid, ep, **kw: "main" if aid == "player_0" else "opponent",
        policies_to_train=["main"],  # Only train main policy
    )
    .callbacks(SelfPlayCallback)
)
```

### Historical Self-Play

Sample opponents from a pool of past checkpoints to prevent forgetting.

```python
import random
from pathlib import Path

class HistoricalSelfPlayCallback:
    """Maintain pool of historical opponents."""

    def __init__(self, checkpoint_dir: str, pool_size: int = 10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.pool_size = pool_size
        self.checkpoints: list[Path] = []

    def on_train_result(self, *, algorithm, result, **kwargs):
        iteration = result["training_iteration"]

        # Save checkpoint periodically
        if iteration % 20 == 0:
            ckpt_path = algorithm.save(str(self.checkpoint_dir / f"iter_{iteration}"))
            self.checkpoints.append(Path(ckpt_path))
            if len(self.checkpoints) > self.pool_size:
                self.checkpoints.pop(0)  # Remove oldest

    def sample_opponent_weights(self, algorithm):
        """Load weights from random historical checkpoint."""
        if not self.checkpoints:
            return algorithm.get_policy("main").get_weights()
        ckpt = random.choice(self.checkpoints)
        # Load and return weights from checkpoint
        return algorithm.from_checkpoint(str(ckpt)).get_policy("main").get_weights()
```

## Population-Based Training for Multi-Agent

Population-Based Training (PBT) evolves a population of agents, combining hyperparameter optimization with self-play diversity.

### PBT Configuration

```python
from ray.tune.schedulers import PopulationBasedTraining

pbt_scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=10,
    quantile_fraction=0.25,  # Bottom 25% copies from top 25%
    resample_probability=0.25,
    hyperparam_mutations={
        "lr": [1e-5, 1e-4, 1e-3],
        "gamma": [0.95, 0.97, 0.99],
        "entropy_coeff": [0.0, 0.01, 0.05],
    },
)
```

### Multi-Agent PBT with Diverse Opponents

Train a population where agents play against each other.

```python
from ray import tune
from ray.tune.tuner import Tuner
from ray.tune.tune_config import TuneConfig

def train_with_population_opponents(config):
    """Training function that samples opponents from population."""
    algo = config["algo_config"].build()

    for _ in range(config["num_iterations"]):
        # Sample opponent from population (via shared storage or ray.get)
        result = algo.train()
        tune.report(**result)

    algo.stop()

tuner = Tuner(
    train_with_population_opponents,
    param_space={
        "algo_config": base_config,
        "num_iterations": 100,
    },
    tune_config=TuneConfig(
        num_samples=8,  # Population size
        scheduler=pbt_scheduler,
        metric="env_runners/episode_reward_mean",
        mode="max",
    ),
)
```

## League Training

League training, popularized by AlphaStar, maintains multiple agent archetypes that play different roles in the training ecosystem.

### League Structure

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable

class AgentType(Enum):
    MAIN_AGENT = "main"           # Primary agents being optimized
    MAIN_EXPLOITER = "exploiter"  # Trained to beat main agents
    LEAGUE_EXPLOITER = "league"   # Trained to beat entire league

@dataclass
class LeagueAgent:
    agent_id: str
    agent_type: AgentType
    policy_weights: dict
    win_rate: float = 0.5
    games_played: int = 0

class League:
    """Manages a population of agents with different training objectives."""

    def __init__(self):
        self.agents: dict[str, LeagueAgent] = {}
        self.matchmaking_history: list[tuple[str, str, float]] = []

    def add_agent(self, agent: LeagueAgent) -> None:
        self.agents[agent.agent_id] = agent

    def sample_opponent(self, player_id: str) -> str:
        """Sample opponent based on agent type and win rates."""
        player = self.agents[player_id]

        if player.agent_type == AgentType.MAIN_AGENT:
            # Play against historical selves and exploiters
            candidates = [
                a for a in self.agents.values()
                if a.agent_id != player_id
            ]
        elif player.agent_type == AgentType.MAIN_EXPLOITER:
            # Focus on current main agents
            candidates = [
                a for a in self.agents.values()
                if a.agent_type == AgentType.MAIN_AGENT
            ]
        else:  # LEAGUE_EXPLOITER
            # Play against entire league
            candidates = [
                a for a in self.agents.values()
                if a.agent_id != player_id
            ]

        # Prioritized Fictitious Self-Play: sample based on win rates
        weights = [1.0 - a.win_rate for a in candidates]  # Prefer harder opponents
        total = sum(weights)
        probs = [w / total for w in weights]

        import random
        return random.choices([a.agent_id for a in candidates], probs)[0]

    def update_ratings(self, player_id: str, opponent_id: str, result: float) -> None:
        """Update win rates after a match. Result: 1.0 = win, 0.5 = draw, 0.0 = loss."""
        player = self.agents[player_id]
        opponent = self.agents[opponent_id]

        # Simple ELO-like update
        player.games_played += 1
        opponent.games_played += 1

        k = 0.1  # Learning rate
        player.win_rate += k * (result - player.win_rate)
        opponent.win_rate += k * ((1 - result) - opponent.win_rate)

        self.matchmaking_history.append((player_id, opponent_id, result))
```

## Emergent Behavior Analysis

Understanding and measuring emergent behaviors in trained multi-agent systems.

### Behavioral Metrics

```python
import numpy as np
from collections import defaultdict

class BehaviorAnalyzer:
    """Analyze emergent behaviors in multi-agent episodes."""

    def __init__(self):
        self.episode_data = []

    def record_step(self, observations, actions, rewards, infos):
        self.episode_data.append({
            "obs": observations,
            "actions": actions,
            "rewards": rewards,
            "infos": infos,
        })

    def compute_coordination_metrics(self) -> dict:
        """Compute metrics indicating coordination quality."""
        metrics = {}

        # Action synchronization: how often agents take similar actions
        action_sequences = defaultdict(list)
        for step in self.episode_data:
            for aid, action in step["actions"].items():
                action_sequences[aid].append(action)

        # Compute pairwise action correlation
        agents = list(action_sequences.keys())
        correlations = []
        for i, a1 in enumerate(agents):
            for a2 in agents[i+1:]:
                seq1, seq2 = action_sequences[a1], action_sequences[a2]
                corr = np.corrcoef(seq1, seq2)[0, 1]
                correlations.append(corr)

        metrics["action_correlation_mean"] = np.mean(correlations)
        metrics["action_correlation_std"] = np.std(correlations)

        # Reward fairness: Gini coefficient of cumulative rewards
        cumulative_rewards = {}
        for step in self.episode_data:
            for aid, reward in step["rewards"].items():
                cumulative_rewards[aid] = cumulative_rewards.get(aid, 0) + reward

        rewards_array = np.array(list(cumulative_rewards.values()))
        metrics["reward_gini"] = self._gini_coefficient(rewards_array)

        return metrics

    def _gini_coefficient(self, values: np.ndarray) -> float:
        """Compute Gini coefficient (0 = perfect equality, 1 = max inequality)."""
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
```

### Visualization Tools

```python
import matplotlib.pyplot as plt

def plot_learning_curves(results: list[dict], policy_ids: list[str]) -> None:
    """Plot per-policy learning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for policy_id in policy_ids:
        rewards = [
            r.get("env_runners", {}).get("policy_reward_mean", {}).get(policy_id, 0)
            for r in results
        ]
        axes[0].plot(rewards, label=policy_id)

    axes[0].set_xlabel("Training Iteration")
    axes[0].set_ylabel("Mean Episode Reward")
    axes[0].legend()
    axes[0].set_title("Per-Policy Rewards")

    # Episode length
    lengths = [r.get("env_runners", {}).get("episode_len_mean", 0) for r in results]
    axes[1].plot(lengths)
    axes[1].set_xlabel("Training Iteration")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_title("Episode Duration")

    plt.tight_layout()
    plt.savefig("learning_curves.png")
```

## Multi-Agent Evaluation Metrics

Comprehensive evaluation requires metrics beyond simple reward.

### Standard Metrics

```python
@dataclass
class MultiAgentEvalMetrics:
    """Comprehensive evaluation metrics for MARL."""

    # Performance
    episode_reward_mean: dict[str, float]
    episode_reward_std: dict[str, float]
    episode_length_mean: float
    win_rate: dict[str, float]  # For competitive settings

    # Efficiency
    sample_efficiency: float  # Reward per environment step
    wall_time_efficiency: float  # Reward per second

    # Coordination (cooperative settings)
    team_reward_mean: float
    coordination_score: float  # Custom metric

    # Robustness
    reward_vs_random_opponent: dict[str, float]
    reward_vs_heuristic_opponent: dict[str, float]

    # Fairness (for teams)
    reward_gini: float
    contribution_variance: float
```

### Evaluation Protocol

```python
def comprehensive_evaluation(
    algorithm,
    num_episodes: int = 100,
    opponent_policies: list[str] = None,
) -> MultiAgentEvalMetrics:
    """Run comprehensive multi-agent evaluation."""

    results = defaultdict(list)

    for episode_idx in range(num_episodes):
        episode_rewards = defaultdict(float)
        episode_length = 0

        # Run episode
        env = algorithm.workers.local_worker().env
        obs, info = env.reset()
        done = {"__all__": False}

        while not done["__all__"]:
            actions = {}
            for aid, ob in obs.items():
                policy_id = algorithm.config.policy_mapping_fn(aid, None)
                actions[aid] = algorithm.compute_single_action(ob, policy_id=policy_id)

            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            done = {k: terminateds.get(k, False) or truncateds.get(k, False)
                    for k in terminateds}

            for aid, r in rewards.items():
                if aid != "__all__":
                    episode_rewards[aid] += r
            episode_length += 1

        # Record results
        for aid, total_reward in episode_rewards.items():
            results[f"reward_{aid}"].append(total_reward)
        results["episode_length"].append(episode_length)

    # Compute summary statistics
    metrics = MultiAgentEvalMetrics(
        episode_reward_mean={
            aid: np.mean(results[f"reward_{aid}"])
            for aid in env.possible_agents
        },
        episode_reward_std={
            aid: np.std(results[f"reward_{aid}"])
            for aid in env.possible_agents
        },
        episode_length_mean=np.mean(results["episode_length"]),
        win_rate={},  # Compute if applicable
        sample_efficiency=0.0,  # Compute from training history
        wall_time_efficiency=0.0,
        team_reward_mean=np.mean([
            sum(results[f"reward_{aid}"][i] for aid in env.possible_agents)
            for i in range(num_episodes)
        ]),
        coordination_score=0.0,  # Custom metric
        reward_vs_random_opponent={},
        reward_vs_heuristic_opponent={},
        reward_gini=0.0,
        contribution_variance=0.0,
    )

    return metrics
```

## Summary

Effective multi-agent coordination in RLlib requires thoughtful selection of:

1. **Training paradigm**: CTDE balances training stability with deployment flexibility.
2. **Parameter sharing**: Match sharing strategy to agent homogeneity and role differentiation.
3. **Communication**: Add explicit channels when implicit coordination is insufficient.
4. **Reward design**: Carefully balance individual and collective incentives.
5. **Self-play structure**: Progress from basic to historical to league training as needed.
6. **Evaluation**: Go beyond reward to measure coordination, robustness, and fairness.

Refer to the main `SKILL.md` for environment setup and basic multi-agent configuration patterns.
