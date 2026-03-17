# Design: ray-ecosystem Plugin

**Date**: 2026-03-17
**Status**: Draft
**Author**: abelyaev-ai

## Summary

A single Claude Code plugin (`ray-ecosystem`) bundling two lean agents and ten
domain skills covering the full Ray stack: Core, Serve, Data, Train, Tune, and
RLlib. Agents act as orchestrators that reference skills for domain knowledge.
Skills are independently invokable by users. All library documentation is
resolved via context7 MCP at runtime.

## Decision Record

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Plugin count | Single `ray-ecosystem` | Ray users work across components; one install covers the natural workflow |
| Agent role | Lean orchestrators | Keeps agent context small; domain knowledge lives in skills |
| Skill count | 10 (5 infra + 5 RL) | Maps 1:1 to distinct Ray concerns (subsystems + operational); no overlap |
| Context7 integration | Skill-level instructions | Context7 is globally available per Constitution Principle I; no plugin-level MCP needed |
| Hooks / Commands | None | No lifecycle automation or slash commands needed for v1 (Principle V) |
| `.mcp.json` | Omitted | Context7 is globally available per Principle I; no plugin-level MCP config needed |

## Plugin Manifest

```json
{
  "name": "ray-ecosystem",
  "description": "Agents and skills for Ray Core, Serve, Data, Train, Tune, and RLlib development",
  "version": "1.0.0",
  "author": {
    "name": "abelyaev-ai"
  },
  "license": "MIT",
  "keywords": ["ray", "distributed-computing", "rllib", "reinforcement-learning", "ml-serving"]
}
```

## Directory Structure

```
ray-ecosystem/
├── .claude-plugin/
│   └── plugin.json
├── agents/
│   ├── ray-expert.md
│   └── ray-rllib-expert.md
├── skills/
│   ├── ray-core-patterns/
│   │   ├── SKILL.md
│   │   └── references/
│   │       └── patterns.md
│   ├── ray-serve-deployment/
│   │   ├── SKILL.md
│   │   └── references/
│   │       └── deployment-guide.md
│   ├── ray-data-train-pipelines/
│   │   ├── SKILL.md
│   │   └── references/
│   │       └── pipeline-patterns.md
│   ├── ray-tune-optimization/
│   │   ├── SKILL.md
│   │   └── references/
│   │       └── search-schedulers.md
│   ├── ray-cluster-config/
│   │   ├── SKILL.md
│   │   └── references/
│   │       └── production-config.md
│   ├── rllib-environment-setup/
│   │   ├── SKILL.md
│   │   └── references/
│   │       └── env-patterns.md
│   ├── rllib-training-pipeline/
│   │   ├── SKILL.md
│   │   └── references/
│   │       └── training-workflows.md
│   ├── rllib-coding-standards/
│   │   ├── SKILL.md
│   │   └── references/
│   │       └── api-migration.md
│   ├── rllib-multi-agent/
│   │   ├── SKILL.md
│   │   └── references/
│   │       └── coordination-patterns.md
│   └── rllib-advanced-techniques/
│       ├── SKILL.md
│       └── references/
│           └── advanced-rl.md
└── README.md
```

## Agents

Both agents share a common 7-step analysis workflow (query breakdown,
context review, component identification, approach planning, domain-specific
considerations, edge cases, collaboration assessment). The ray-rllib-expert
additionally uses a 4-step implementation workflow (Context, Design,
Implement, Validate) that wraps around the analysis. This is an
orchestration layer, not domain knowledge.

### Agent File Template

```markdown
---
description: >
  <One-paragraph role description with expertise scope.
   Must be distinguishable from the other agent.>
---

## Analysis Workflow
<7-step systematic analysis — shared structure, domain-specific wording>

## Context7 Integration
Before implementation, resolve current docs:
1. Call `resolve-library-id` for "ray"
2. Call `query-docs` for the relevant subsystem

## Skill References
<List of skills this agent invokes, with when-to-use guidance>

## Collaboration
<When to delegate to the other agent>
```

### ray-expert

**Role**: Lean orchestrator for Ray infrastructure (Core, Serve, Data, Train,
Tune) and cluster architecture.

**What stays in the agent**:
- Persona and expertise scope declaration
- 7-step systematic analysis workflow (query breakdown, context review,
  component identification, approach planning, Ray-specific considerations,
  edge cases, collaboration assessment)
- Collaboration routing: delegates RL to ray-rllib-expert
- Context7 instruction: resolve Ray docs before implementation
- Skill references pointing to the 5 infrastructure skills

**What moves to skills**:
- All bullet-point domain knowledge about actors, tasks, remote functions
- Ray Serve deployment patterns and strategies
- Data/Train pipeline details
- Tune search algorithms and schedulers
- Cluster management and autoscaling specifics
- Hardcoded documentation URLs (replaced by context7)

**Target size**: ~50 lines (down from 114)

### ray-rllib-expert

**Role**: Lean orchestrator for RLlib and distributed reinforcement learning,
new API stack (2.53+).

**What stays in the agent**:
- Persona and expertise scope declaration
- 4-step workflow: Context, Design, Implement, Validate
- 7-step analysis instructions
- Quality gate checklist (orchestration concern — "did I verify everything?")
- Collaboration routing: delegates infra to ray-expert
- Context7 instruction: resolve RLlib docs before implementation
- Skill references pointing to the 5 RL skills

**What moves to skills**:
- Algorithm listings and details (PPO, SAC, DQN, etc.)
- Environment and policy specifics (Gymnasium 5-tuple, vectorization)
- Coding standards and anti-patterns (new API requirements, forbidden patterns)
- Multi-agent RL patterns
- Advanced techniques (offline RL, hierarchical, meta-RL)
- Type safety rules and testing patterns
- Hardcoded documentation URLs (replaced by context7)

**Target size**: ~60 lines (down from 212)

## Skills

All skills follow the same structural pattern:

```markdown
---
name: <skill-name>
description: >
  This skill should be used when the user asks to "<trigger phrase 1>",
  "<trigger phrase 2>", "<trigger phrase 3>", or <broader context>.
---

## Purpose
<2-3 sentences: what this skill provides>

## Prerequisites
Resolve current API documentation before proceeding:
1. Call `resolve-library-id` for "ray" (or "rllib")
2. Call `query-docs` for <specific topic>

## Core Workflow
<Step-by-step guidance in imperative form, typically 800-1200 words;
 prioritize completeness over word count>

## Common Pitfalls
<3-5 bullet points of non-obvious mistakes>

## Additional Resources
### Reference Files
- **`references/<file>.md`** — <what it contains>
```

### Skill Descriptions (trigger lines)

| Skill | Description |
|-------|-------------|
| `ray-core-patterns` | This skill should be used when the user asks to "create a Ray actor", "set up remote tasks", "use the object store", "handle Ray fault tolerance", or works with Ray Core distributed primitives. |
| `ray-serve-deployment` | This skill should be used when the user asks to "deploy a model with Ray Serve", "set up HTTP endpoints", "configure A/B testing", "scale a serving deployment", or works with Ray Serve. |
| `ray-data-train-pipelines` | This skill should be used when the user asks to "build a data pipeline with Ray", "set up distributed training", "preprocess data with Ray Data", or works with Ray Data or Ray Train. |
| `ray-tune-optimization` | This skill should be used when the user asks to "tune hyperparameters with Ray", "configure a search algorithm", "set up a Tune scheduler", or works with Ray Tune experiments. |
| `ray-cluster-config` | This skill should be used when the user asks to "set up a Ray cluster", "configure autoscaling", "allocate cluster resources", "deploy Ray to production", or works with Ray cluster management. |
| `rllib-environment-setup` | This skill should be used when the user asks to "create a custom RL environment", "scaffold a Gymnasium env", "define observation and action spaces", or sets up environments for RLlib. |
| `rllib-training-pipeline` | This skill should be used when the user asks to "train an RLlib algorithm", "set up AlgorithmConfig", "checkpoint a training run", "evaluate an RL policy", or builds RLlib training loops. |
| `rllib-coding-standards` | This skill should be used when the user asks about "RLlib new API", "RLlib anti-patterns", "migrate from old RLlib API", "RLlib type safety", or needs RLlib coding conventions (2.53+). |
| `rllib-multi-agent` | This skill should be used when the user asks to "set up multi-agent RL", "configure policy mapping", "coordinate multiple agents", "shape rewards for MARL", or works with RLlib multi-agent. |
| `rllib-advanced-techniques` | This skill should be used when the user asks about "offline RL", "behavioral cloning", "hierarchical RL", "meta-RL", "curiosity-driven exploration", or implements advanced RLlib techniques. |

### Skill Content Sources

Source material for skills comes from two existing agent files at
`~/.claude/agents/ray-expert.md` (114 lines) and
`~/.claude/agents/ray-rllib-expert.md` (212 lines). These files will be
read during implementation but are **not** committed to this repo — they
serve as starting points only.

| Skill | Source agent | Topics extracted |
|-------|-------------|-----------------|
| `ray-core-patterns` | ray-expert | Actors, tasks, remote functions, resource allocation, fault tolerance, performance |
| `ray-serve-deployment` | ray-expert | Deployment patterns, HTTP endpoints, model versioning, A/B testing, scaling |
| `ray-data-train-pipelines` | ray-expert | Data preprocessing, distributed training, framework integration |
| `ray-tune-optimization` | ray-expert | Search algorithms, schedulers, experiment config, distributed HPO |
| `ray-cluster-config` | ray-expert | Cluster management, autoscaling, production deployment |
| `rllib-environment-setup` | ray-rllib-expert | Gymnasium 5-tuple, custom envs, type hints, vectorization |
| `rllib-training-pipeline` | ray-rllib-expert | Algorithms, AlgorithmConfig builders, training, callbacks |
| `rllib-coding-standards` | ray-rllib-expert | New API requirements, type safety, error handling, anti-patterns |
| `rllib-multi-agent` | ray-rllib-expert | Centralized/decentralized, policy mapping, coordination |
| `rllib-advanced-techniques` | ray-rllib-expert | Offline RL, hierarchical, meta-RL, exploration |

Content in `SKILL.md` bodies will be **expanded beyond the original agent
bullets** using context7 to fetch current Ray documentation. The original
agents had terse listings; skills will provide actionable step-by-step
workflows authored fresh with up-to-date API details.

## Data Flow

```
User query about Ray
        │
        ▼
Claude Code routes to agent
(based on agent description matching)
        │
        ├─ Infrastructure query ──▶ ray-expert agent
        │                              │
        │                              ├─ Invokes relevant skill(s)
        │                              ├─ Calls context7 for docs
        │                              └─ Returns solution
        │
        └─ RL/RLlib query ────────▶ ray-rllib-expert agent
                                       │
                                       ├─ Invokes relevant skill(s)
                                       ├─ Calls context7 for docs
                                       └─ Returns solution

User can also invoke skills directly:
  "Help me create a Ray actor" ──▶ ray-core-patterns skill triggers
```

## Error Handling

- **Context7 unavailable**: Skills instruct to proceed with training knowledge
  but flag that docs may be stale. No hard failure.
- **Wrong agent routed**: Agent descriptions are non-overlapping. ray-expert
  handles infra, ray-rllib-expert handles RL. If misrouted, the agent
  redirects to its counterpart via collaboration routing.
- **Skill not triggered**: Agent system prompts explicitly list which skills to
  invoke for which topics, acting as a fallback routing mechanism.

## Testing Strategy

1. **Structure validation**: Run `plugin-validator` agent against the plugin
2. **Skill triggering**: Test each skill's description against sample queries
   to verify correct activation
3. **Agent leanness**: Verify agents contain no domain knowledge duplicated in
   skills
4. **Context7 integration**: Confirm `resolve-library-id` and `query-docs`
   calls work for Ray/RLlib
5. **Local testing**: `claude --plugin-dir ./ray-ecosystem` with representative
   queries across all Ray subsystems
6. **Description scan test**: Read all 10 skill descriptions in sequence and
   confirm each is immediately distinguishable

## Marketplace Readiness

Per Constitution Principle IV:
- Valid `plugin.json` with all recommended fields
- README.md with purpose, installation, component list, usage examples
- Self-contained directory — no external dependencies beyond context7
- Semantic versioning starting at 1.0.0
- No hardcoded paths (use `${CLAUDE_PLUGIN_ROOT}` where needed)

## Success Criteria

- **SC-001**: Plugin installs and loads via `claude --plugin-dir ./ray-ecosystem`
- **SC-002**: Both agents trigger correctly on domain-appropriate queries
- **SC-003**: All 10 skills trigger on their documented trigger phrases
- **SC-004**: Agent context stays under 60 lines each (no domain duplication)
- **SC-005**: `plugin-validator` passes with no errors
- **SC-006**: A user scanning the 10 skill descriptions can identify the right
  skill for any Ray task within 5 seconds
