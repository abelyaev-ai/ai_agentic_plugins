# ray-ecosystem

Claude Code plugin for the Ray distributed computing ecosystem. Provides
specialized agents and skills covering Ray Core, Ray Serve, Ray Data,
Ray Train, Ray Tune, and RLlib.

## Installation

```
/plugin install ray-ecosystem@ray-ecosystem
```

Or test locally during development:

```bash
claude --plugin-dir ./ray-ecosystem
```

## Prerequisites

- **context7 MCP server** must be available in the environment for
  up-to-date API documentation resolution.

## Components

### Agents

| Agent | Domain |
|-------|--------|
| `ray-expert` | Ray infrastructure orchestrator: Core, Serve, Data, Train, Tune, cluster architecture |
| `ray-rllib-expert` | RLlib orchestrator: RL algorithms, environments, multi-agent, new API stack (2.53+) |

Agents are lean orchestrators that delegate domain knowledge to skills
and resolve documentation via context7.

### Skills

#### Infrastructure (invoked by ray-expert)

| Skill | Purpose |
|-------|---------|
| `ray-core-patterns` | Actors, tasks, remote functions, object store, fault tolerance |
| `ray-serve-deployment` | Model serving, HTTP endpoints, A/B testing, scaling |
| `ray-data-train-pipelines` | Data preprocessing, distributed training workflows |
| `ray-tune-optimization` | Hyperparameter search algorithms, schedulers, experiments |
| `ray-cluster-config` | Cluster setup, autoscaling, production deployment |

#### RLlib (invoked by ray-rllib-expert)

| Skill | Purpose |
|-------|---------|
| `rllib-environment-setup` | Custom Gymnasium environments, spaces, 5-tuple compliance |
| `rllib-training-pipeline` | AlgorithmConfig, training loops, checkpointing, evaluation |
| `rllib-coding-standards` | New API (2.53+) requirements, anti-patterns, migration guide |
| `rllib-multi-agent` | Multi-agent coordination, policy mapping, reward shaping |
| `rllib-advanced-techniques` | Offline RL, hierarchical RL, meta-RL, curiosity exploration |

Skills can also be triggered directly by user queries without going
through an agent.

## Usage Examples

```
"Help me create a Ray actor for a game server"
  -> triggers ray-core-patterns skill

"Deploy my model with Ray Serve behind an HTTP endpoint"
  -> triggers ray-serve-deployment skill

"Set up distributed training with PyTorch and Ray Train"
  -> triggers ray-data-train-pipelines skill

"Train a PPO agent on my custom environment"
  -> triggers rllib-training-pipeline skill

"Migrate my RLlib code from the old API to the new stack"
  -> triggers rllib-coding-standards skill

"Set up multi-agent reinforcement learning with shared policies"
  -> triggers rllib-multi-agent skill
```

## License

MIT
