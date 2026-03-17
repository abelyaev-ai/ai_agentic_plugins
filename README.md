# ai_agentic_plugins

Claude Code plugin marketplace for AI/ML ecosystems.

## Installation

Add this marketplace to Claude Code:

```
/plugin marketplace add abelyaev-ai/ai_agentic_plugins
```

Then install plugins:

```
/plugin install ray-ecosystem@ai-agentic-plugins
```

## Available Plugins

| Plugin | Version | Description |
|--------|---------|-------------|
| [ray-ecosystem](./ray-ecosystem/) | 1.0.0 | Agents and skills for Ray Core, Serve, Data, Train, Tune, and RLlib |

## Plugin Details

### ray-ecosystem

2 specialized agents + 10 domain skills covering the full Ray distributed
computing stack. Agents act as lean orchestrators; domain knowledge lives
in skills with up-to-date documentation resolved via context7 MCP.

**Agents:** `ray-expert` (infrastructure), `ray-rllib-expert` (reinforcement learning)

**Skills:** ray-core-patterns, ray-serve-deployment, ray-data-train-pipelines,
ray-tune-optimization, ray-cluster-config, rllib-environment-setup,
rllib-training-pipeline, rllib-coding-standards, rllib-multi-agent,
rllib-advanced-techniques

See [ray-ecosystem/README.md](./ray-ecosystem/README.md) for full details.

## License

MIT
