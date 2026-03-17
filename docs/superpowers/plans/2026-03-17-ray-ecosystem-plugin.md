# ray-ecosystem Plugin Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Claude Code plugin bundling 2 lean agents and 10 domain skills for the Ray ecosystem.

**Architecture:** Single plugin (`ray-ecosystem`) with agents as orchestrators referencing skills for domain knowledge. Skills authored using context7 for up-to-date API docs. Source material extracted from existing `~/.claude/agents/ray-expert.md` and `~/.claude/agents/ray-rllib-expert.md`.

**Tech Stack:** Markdown (SKILL.md, agent .md files), JSON (plugin.json)

**Spec:** `docs/superpowers/specs/2026-03-17-ray-ecosystem-plugin-design.md`

**Prerequisites:** Source agent files must exist:
```bash
test -f ~/.claude/agents/ray-expert.md && test -f ~/.claude/agents/ray-rllib-expert.md && echo "Source agents found" || echo "WARNING: source agents not found — agents will be authored from context7 docs only"
```

---

## Task 1: Scaffold plugin directory and manifest

**Files:**
- Create: `ray-ecosystem/.claude-plugin/plugin.json`
- Create: all skill subdirectories and `references/` dirs

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p ray-ecosystem/.claude-plugin
mkdir -p ray-ecosystem/agents
mkdir -p ray-ecosystem/skills/ray-core-patterns/references
mkdir -p ray-ecosystem/skills/ray-serve-deployment/references
mkdir -p ray-ecosystem/skills/ray-data-train-pipelines/references
mkdir -p ray-ecosystem/skills/ray-tune-optimization/references
mkdir -p ray-ecosystem/skills/ray-cluster-config/references
mkdir -p ray-ecosystem/skills/rllib-environment-setup/references
mkdir -p ray-ecosystem/skills/rllib-training-pipeline/references
mkdir -p ray-ecosystem/skills/rllib-coding-standards/references
mkdir -p ray-ecosystem/skills/rllib-multi-agent/references
mkdir -p ray-ecosystem/skills/rllib-advanced-techniques/references
```

- [ ] **Step 2: Write plugin.json**

Create `ray-ecosystem/.claude-plugin/plugin.json`:

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

- [ ] **Step 3: Verify structure**

```bash
find ray-ecosystem -type d | sort
```

Expected: 24 directories (root + .claude-plugin + agents + skills + 10 skill dirs + 10 references dirs)

- [ ] **Step 4: Commit**

```bash
git add ray-ecosystem/.claude-plugin/plugin.json
git commit -m "feat: scaffold ray-ecosystem plugin structure and manifest"
```

---

## Task 2: Write ray-expert agent

**Files:**
- Read: `~/.claude/agents/ray-expert.md` (source material)
- Create: `ray-ecosystem/agents/ray-expert.md`

The agent keeps: persona, 7-step analysis workflow, context7 instruction,
skill references, collaboration routing. Everything else moves to skills.

- [ ] **Step 1: Read source agent**

Read `~/.claude/agents/ray-expert.md` to understand the analysis workflow
structure and collaboration patterns.

- [ ] **Step 2: Write lean agent**

Create `ray-ecosystem/agents/ray-expert.md` with:
- `description` frontmatter (one paragraph, distinguishable from rllib-expert)
- 7-step analysis workflow (rewritten for lean orchestrator role)
- Context7 integration section
- Skill references table: which skill to invoke for which topic
- Collaboration section: when to delegate to ray-rllib-expert

Target: ~50 lines. No domain knowledge bullets. No hardcoded doc URLs.

- [ ] **Step 3: Verify line count and no domain leakage**

```bash
wc -l ray-ecosystem/agents/ray-expert.md
```

Expected: 40-60 lines. Grep for domain content that should be in skills:

```bash
grep -i "actor\|remote function\|object store\|deployment pattern\|A/B test\|preprocessing\|search algorithm\|scheduler\|autoscal" ray-ecosystem/agents/ray-expert.md
```

Expected: only references in the skill references table, not standalone knowledge.

- [ ] **Step 4: Commit**

```bash
git add ray-ecosystem/agents/ray-expert.md
git commit -m "feat: add lean ray-expert agent with skill references"
```

---

## Task 3: Write ray-rllib-expert agent

**Files:**
- Read: `~/.claude/agents/ray-rllib-expert.md` (source material)
- Create: `ray-ecosystem/agents/ray-rllib-expert.md`

Keeps: persona, 4-step workflow (Context/Design/Implement/Validate),
7-step analysis, quality gate checklist, context7 instruction,
skill references, collaboration routing.

- [ ] **Step 1: Read source agent**

Read `~/.claude/agents/ray-rllib-expert.md` to understand the analysis
workflow, quality gate checklist, and coding standards structure.

- [ ] **Step 2: Write lean agent**

Create `ray-ecosystem/agents/ray-rllib-expert.md` with:
- `description` frontmatter (one paragraph, distinguishable from ray-expert)
- 4-step implementation workflow
- 7-step analysis instructions (RLlib-specific wording)
- Quality gate checklist (kept — orchestration, not domain knowledge)
- Context7 integration section
- Skill references table
- Collaboration section: when to delegate to ray-expert

Target: ~60 lines. Coding standards, anti-patterns, algorithm details,
env specifics all moved to skills.

- [ ] **Step 3: Verify line count and no domain leakage**

```bash
wc -l ray-ecosystem/agents/ray-rllib-expert.md
```

Expected: 50-70 lines. Grep for domain content that should be in skills:

```bash
grep -i "PPOTrainer\|ModelV2\|RolloutWorker\|num_workers\|train_batch_size[^_]\|Gymnasium 5-tuple\|behavioral cloning\|curiosity" ray-ecosystem/agents/ray-rllib-expert.md
```

Expected: zero matches (this content lives in skills now).

- [ ] **Step 4: Commit**

```bash
git add ray-ecosystem/agents/ray-rllib-expert.md
git commit -m "feat: add lean ray-rllib-expert agent with skill references"
```

---

## Task 4: Write ray-core-patterns skill [P]

**Files:**
- Create: `ray-ecosystem/skills/ray-core-patterns/SKILL.md`
- Create: `ray-ecosystem/skills/ray-core-patterns/references/patterns.md`

Source: ray-expert "Ray Core & Distributed Computing" section.
Expand using context7 (`resolve-library-id` for "ray", `query-docs` for
"core actors tasks remote functions").

- [ ] **Step 1: Resolve context7 docs**

Call `resolve-library-id` for "ray", then `query-docs` for Ray Core
actors, tasks, remote functions, object store, fault tolerance.

- [ ] **Step 2: Write SKILL.md**

Frontmatter: name `ray-core-patterns`, third-person description with
trigger phrases ("create a Ray actor", "set up remote tasks", etc.).

Body (imperative form, ~1500 words):
- Purpose (2-3 sentences)
- Prerequisites (context7 resolution steps)
- Core Workflow: actors, tasks, remote functions, object store usage,
  resource requests, fault tolerance patterns
- Common Pitfalls (3-5 bullets)
- Additional Resources pointer to `references/patterns.md`

- [ ] **Step 3: Write references/patterns.md**

Detailed patterns (~2000-3000 words):
- Actor lifecycle management patterns
- Task DAG composition
- Object store best practices and serialization
- Resource allocation and scheduling strategies
- Fault tolerance: retries, lineage reconstruction, placement groups
- Performance debugging with Ray Dashboard

- [ ] **Step 4: Verify SKILL.md word count and style**

```bash
wc -w ray-ecosystem/skills/ray-core-patterns/SKILL.md
```

Expected: 1200-2500 words total (including frontmatter).
Check for second-person ("you should") — should be zero:

```bash
grep -ci "you should\|you need\|you can\|you must" ray-ecosystem/skills/ray-core-patterns/SKILL.md
```

- [ ] **Step 5: Commit**

```bash
git add ray-ecosystem/skills/ray-core-patterns/
git commit -m "feat: add ray-core-patterns skill with reference docs"
```

---

## Task 5: Write ray-serve-deployment skill [P]

**Files:**
- Create: `ray-ecosystem/skills/ray-serve-deployment/SKILL.md`
- Create: `ray-ecosystem/skills/ray-serve-deployment/references/deployment-guide.md`

Source: ray-expert "Ray Serve" section.
Expand using context7 for Ray Serve deployment, HTTP, scaling.

- [ ] **Step 1: Resolve context7 docs for Ray Serve**

- [ ] **Step 2: Write SKILL.md**

Triggers: "deploy a model with Ray Serve", "set up HTTP endpoints",
"configure A/B testing", "scale a serving deployment".

Body: deployment creation, HTTP endpoint binding, model composition,
scaling policies, A/B testing with traffic splitting.

- [ ] **Step 3: Write references/deployment-guide.md**

Detailed: deployment graph patterns, FastAPI integration, batching,
model multiplexing, production deployment with Kubernetes.

- [ ] **Step 4: Verify word count and style**

```bash
wc -w ray-ecosystem/skills/ray-serve-deployment/SKILL.md
grep -ci "you should\|you need\|you can\|you must" ray-ecosystem/skills/ray-serve-deployment/SKILL.md
```

Expected: 1200-2500 words, zero second-person matches.

- [ ] **Step 5: Commit**

```bash
git add ray-ecosystem/skills/ray-serve-deployment/
git commit -m "feat: add ray-serve-deployment skill with reference docs"
```

---

## Task 6: Write ray-data-train-pipelines skill [P]

**Files:**
- Create: `ray-ecosystem/skills/ray-data-train-pipelines/SKILL.md`
- Create: `ray-ecosystem/skills/ray-data-train-pipelines/references/pipeline-patterns.md`

Source: ray-expert "Ray Data & Train" section.
Expand using context7 for Ray Data and Ray Train APIs.

- [ ] **Step 1: Resolve context7 docs for Ray Data and Ray Train**

- [ ] **Step 2: Write SKILL.md**

Triggers: "build a data pipeline with Ray", "set up distributed training",
"preprocess data with Ray Data".

Body: Dataset creation, transformations, reading/writing data sources,
Ray Train integration with PyTorch/Lightning/HuggingFace, scaling
training across workers.

- [ ] **Step 3: Write references/pipeline-patterns.md**

Detailed: data loading patterns, windowed vs batch transforms,
GPU preprocessing, distributed training strategies (DDP, FSDP),
checkpointing and fault tolerance in training.

- [ ] **Step 4: Verify word count and style**

```bash
wc -w ray-ecosystem/skills/ray-data-train-pipelines/SKILL.md
grep -ci "you should\|you need\|you can\|you must" ray-ecosystem/skills/ray-data-train-pipelines/SKILL.md
```

Expected: 1200-2500 words, zero second-person matches.

- [ ] **Step 5: Commit**

```bash
git add ray-ecosystem/skills/ray-data-train-pipelines/
git commit -m "feat: add ray-data-train-pipelines skill with reference docs"
```

---

## Task 7: Write ray-tune-optimization skill [P]

**Files:**
- Create: `ray-ecosystem/skills/ray-tune-optimization/SKILL.md`
- Create: `ray-ecosystem/skills/ray-tune-optimization/references/search-schedulers.md`

Source: ray-expert "Ray Tune" section.
Expand using context7 for Tune search algorithms and schedulers.

- [ ] **Step 1: Resolve context7 docs for Ray Tune**

- [ ] **Step 2: Write SKILL.md**

Triggers: "tune hyperparameters with Ray", "configure a search algorithm",
"set up a Tune scheduler".

Body: Tuner/TuneConfig setup, search space definition, search algorithms
(Optuna, BayesOpt, HyperOpt), schedulers (ASHA, PBT, BOHB),
result analysis and best trial extraction.

- [ ] **Step 3: Write references/search-schedulers.md**

Detailed: algorithm comparison table, scheduler selection guide,
distributed Tune with resources, integration with Ray Train,
custom stopper and reporter patterns.

- [ ] **Step 4: Verify word count and style**

```bash
wc -w ray-ecosystem/skills/ray-tune-optimization/SKILL.md
grep -ci "you should\|you need\|you can\|you must" ray-ecosystem/skills/ray-tune-optimization/SKILL.md
```

Expected: 1200-2500 words, zero second-person matches.

- [ ] **Step 5: Commit**

```bash
git add ray-ecosystem/skills/ray-tune-optimization/
git commit -m "feat: add ray-tune-optimization skill with reference docs"
```

---

## Task 8: Write ray-cluster-config skill [P]

**Files:**
- Create: `ray-ecosystem/skills/ray-cluster-config/SKILL.md`
- Create: `ray-ecosystem/skills/ray-cluster-config/references/production-config.md`

Source: ray-expert cluster management content (expanded — thin in source).
Expand using context7 for Ray cluster setup, autoscaling, KubeRay.

- [ ] **Step 1: Resolve context7 docs for Ray cluster management**

- [ ] **Step 2: Write SKILL.md**

Triggers: "set up a Ray cluster", "configure autoscaling",
"allocate cluster resources", "deploy Ray to production".

Body: ray.init() options, cluster YAML config, head/worker node setup,
autoscaler configuration, resource specification (CPU/GPU/custom),
runtime environment packaging.

- [ ] **Step 3: Write references/production-config.md**

Detailed: KubeRay operator setup, Helm chart configuration,
cloud-specific deployment (AWS/GCP/Azure), monitoring with
Prometheus/Grafana, logging and observability, security configuration.

- [ ] **Step 4: Verify word count and style**

```bash
wc -w ray-ecosystem/skills/ray-cluster-config/SKILL.md
grep -ci "you should\|you need\|you can\|you must" ray-ecosystem/skills/ray-cluster-config/SKILL.md
```

Expected: 1200-2500 words, zero second-person matches.

- [ ] **Step 5: Commit**

```bash
git add ray-ecosystem/skills/ray-cluster-config/
git commit -m "feat: add ray-cluster-config skill with reference docs"
```

---

## Task 9: Write rllib-environment-setup skill [P]

**Files:**
- Create: `ray-ecosystem/skills/rllib-environment-setup/SKILL.md`
- Create: `ray-ecosystem/skills/rllib-environment-setup/references/env-patterns.md`

Source: ray-rllib-expert "Environments & Policies" section.
Expand using context7 for Gymnasium env creation and RLlib env registration.

- [ ] **Step 1: Resolve context7 docs for RLlib environments**

- [ ] **Step 2: Write SKILL.md**

Triggers: "create a custom RL environment", "scaffold a Gymnasium env",
"define observation and action spaces".

Body: Gymnasium env class scaffold (with 5-tuple), observation/action
space design (Box, Discrete, Dict, MultiDiscrete), reset/step
implementation, env registration, vectorized env support, type hints.

- [ ] **Step 3: Write references/env-patterns.md**

Detailed: multi-agent env patterns (MultiAgentEnv), external env
connectors, env wrapper patterns, observation normalization,
action masking, env validation and debugging, testing fixtures
for env correctness.

- [ ] **Step 4: Verify word count and style**

```bash
wc -w ray-ecosystem/skills/rllib-environment-setup/SKILL.md
grep -ci "you should\|you need\|you can\|you must" ray-ecosystem/skills/rllib-environment-setup/SKILL.md
```

Expected: 1200-2500 words, zero second-person matches.

- [ ] **Step 5: Commit**

```bash
git add ray-ecosystem/skills/rllib-environment-setup/
git commit -m "feat: add rllib-environment-setup skill with reference docs"
```

---

## Task 10: Write rllib-training-pipeline skill [P]

**Files:**
- Create: `ray-ecosystem/skills/rllib-training-pipeline/SKILL.md`
- Create: `ray-ecosystem/skills/rllib-training-pipeline/references/training-workflows.md`

Source: ray-rllib-expert "Algorithms & Training" section.
Expand using context7 for AlgorithmConfig, training loop, checkpointing.

- [ ] **Step 1: Resolve context7 docs for RLlib training**

- [ ] **Step 2: Write SKILL.md**

Triggers: "train an RLlib algorithm", "set up AlgorithmConfig",
"checkpoint a training run", "evaluate an RL policy".

Body: AlgorithmConfig builder pattern (`.environment()`, `.training()`,
`.env_runners()`, `.learners()`), `.build()`, `algo.train()` loop,
checkpoint save/restore, evaluation with `algo.evaluate()`,
stopping criteria, resource specification.

- [ ] **Step 3: Write references/training-workflows.md**

Detailed: algorithm-specific config examples (PPO, SAC, DQN),
curriculum learning setup, custom callbacks, training metrics
interpretation, checkpoint management patterns, training
resumption from checkpoint, distributed training topology.

- [ ] **Step 4: Verify word count and style**

```bash
wc -w ray-ecosystem/skills/rllib-training-pipeline/SKILL.md
grep -ci "you should\|you need\|you can\|you must" ray-ecosystem/skills/rllib-training-pipeline/SKILL.md
```

Expected: 1200-2500 words, zero second-person matches.

- [ ] **Step 5: Commit**

```bash
git add ray-ecosystem/skills/rllib-training-pipeline/
git commit -m "feat: add rllib-training-pipeline skill with reference docs"
```

---

## Task 11: Write rllib-coding-standards skill [P]

**Files:**
- Create: `ray-ecosystem/skills/rllib-coding-standards/SKILL.md`
- Create: `ray-ecosystem/skills/rllib-coding-standards/references/api-migration.md`

Source: ray-rllib-expert "Coding Standards" and "Anti-Patterns" sections.
This is the most prescriptive skill — it moves the bulk of the rllib agent's
standards content.

- [ ] **Step 1: Resolve context7 docs for RLlib new API stack**

Call `resolve-library-id` for "ray", then `query-docs` for RLlib
AlgorithmConfig, new API stack, EnvRunner, RLModule migration.

- [ ] **Step 2: Write SKILL.md**

Triggers: "RLlib new API", "RLlib anti-patterns", "migrate from old
RLlib API", "RLlib type safety".

Body: New API requirements (2.53+), AlgorithmConfig builders,
EnvRunner terminology, Learner-centric batching, PyTorch-only,
type safety rules (full signatures, NDArray, Optional),
error handling (EAFP, chain exceptions, cleanup in finally),
testing patterns (pytest + Ray, fixtures, parametrize).

Anti-patterns section: forbidden old API classes/params, general
Python anti-patterns in RLlib context.

- [ ] **Step 3: Write references/api-migration.md**

Detailed migration guide:
- Old → new API mapping table (PPOTrainer → PPOConfig().build(), etc.)
- num_workers → num_env_runners migration
- ModelV2 → RLModule migration steps
- Dict config → AlgorithmConfig builder migration
- TensorFlow → PyTorch migration notes
- Version compatibility matrix

- [ ] **Step 4: Verify word count and style**

```bash
wc -w ray-ecosystem/skills/rllib-coding-standards/SKILL.md
grep -ci "you should\|you need\|you can\|you must" ray-ecosystem/skills/rllib-coding-standards/SKILL.md
```

Expected: 1200-2500 words, zero second-person matches.

- [ ] **Step 5: Commit**

```bash
git add ray-ecosystem/skills/rllib-coding-standards/
git commit -m "feat: add rllib-coding-standards skill with migration guide"
```

---

## Task 12: Write rllib-multi-agent skill [P]

**Files:**
- Create: `ray-ecosystem/skills/rllib-multi-agent/SKILL.md`
- Create: `ray-ecosystem/skills/rllib-multi-agent/references/coordination-patterns.md`

Source: ray-rllib-expert "Multi-Agent RL" section.
Expand using context7 for MultiAgentEnv, policy mapping.

- [ ] **Step 1: Resolve context7 docs for RLlib multi-agent**

- [ ] **Step 2: Write SKILL.md**

Triggers: "set up multi-agent RL", "configure policy mapping",
"coordinate multiple agents", "shape rewards for MARL".

Body: MultiAgentEnv setup, policy mapping function, multiple
policies with different algorithms, agent grouping, shared vs
independent policies, observation/action space per agent.

- [ ] **Step 3: Write references/coordination-patterns.md**

Detailed: centralized training decentralized execution (CTDE),
parameter sharing strategies, communication channels between agents,
reward shaping for cooperation/competition, emergent behavior
analysis, multi-agent evaluation metrics.

- [ ] **Step 4: Verify word count and style**

```bash
wc -w ray-ecosystem/skills/rllib-multi-agent/SKILL.md
grep -ci "you should\|you need\|you can\|you must" ray-ecosystem/skills/rllib-multi-agent/SKILL.md
```

Expected: 1200-2500 words, zero second-person matches.

- [ ] **Step 5: Commit**

```bash
git add ray-ecosystem/skills/rllib-multi-agent/
git commit -m "feat: add rllib-multi-agent skill with coordination patterns"
```

---

## Task 13: Write rllib-advanced-techniques skill [P]

**Files:**
- Create: `ray-ecosystem/skills/rllib-advanced-techniques/SKILL.md`
- Create: `ray-ecosystem/skills/rllib-advanced-techniques/references/advanced-rl.md`

Source: ray-rllib-expert "Advanced Techniques" section.
Expand using context7 for offline RL, hierarchical RL, exploration.

- [ ] **Step 1: Resolve context7 docs for advanced RLlib techniques**

- [ ] **Step 2: Write SKILL.md**

Triggers: "offline RL", "behavioral cloning", "hierarchical RL",
"meta-RL", "curiosity-driven exploration".

Body: offline RL with dataset integration, behavioral cloning setup,
hierarchical RL with option framework, meta-RL with task
distributions, curiosity-driven and intrinsic motivation exploration,
RLModule customization for advanced architectures.

- [ ] **Step 3: Write references/advanced-rl.md**

Detailed: offline RL data format requirements, CQL/IQL algorithm configs,
hierarchical policy architecture, meta-learning training loops,
exploration strategy comparison, custom RLModule implementation
for advanced architectures.

- [ ] **Step 4: Verify word count and style**

```bash
wc -w ray-ecosystem/skills/rllib-advanced-techniques/SKILL.md
grep -ci "you should\|you need\|you can\|you must" ray-ecosystem/skills/rllib-advanced-techniques/SKILL.md
```

Expected: 1200-2500 words, zero second-person matches.

- [ ] **Step 5: Commit**

```bash
git add ray-ecosystem/skills/rllib-advanced-techniques/
git commit -m "feat: add rllib-advanced-techniques skill with advanced RL guide"
```

---

## Task 14: Write README.md

**Files:**
- Create: `ray-ecosystem/README.md`

- [ ] **Step 1: Write README**

Include per Constitution Principle IV:
- Plugin purpose and scope
- Installation: `/plugin install ray-ecosystem@ray-ecosystem`
- Component list: 2 agents, 10 skills with one-line descriptions
- Usage examples: sample queries that trigger each agent/skill
- Prerequisites: context7 MCP server must be available
- License: MIT

- [ ] **Step 2: Commit**

```bash
git add ray-ecosystem/README.md
git commit -m "docs: add ray-ecosystem plugin README"
```

---

## Task 15: Validate plugin

- [ ] **Step 1: Run plugin-validator**

Use the `plugin-validator` agent against `ray-ecosystem/` to check:
- Valid plugin.json manifest
- All skills have SKILL.md with proper frontmatter
- Agent files have proper frontmatter
- All referenced files exist

- [ ] **Step 2: Check skill descriptions are distinguishable**

Read all 10 SKILL.md frontmatter descriptions in sequence. Verify
each is unique and non-overlapping.

- [ ] **Step 3: Verify agent leanness**

```bash
wc -l ray-ecosystem/agents/ray-expert.md ray-ecosystem/agents/ray-rllib-expert.md
```

Expected: ray-expert ~50 lines, ray-rllib-expert ~60 lines.

- [ ] **Step 4: Test locally**

```bash
claude --plugin-dir ./ray-ecosystem
```

Test with sample queries:
- "Help me create a Ray actor" → should trigger ray-core-patterns skill
- "Set up distributed training" → should trigger ray-data-train-pipelines
- "Train a PPO agent" → should trigger rllib-training-pipeline

- [ ] **Step 5: Success criteria checklist**

Verify against spec success criteria:
- SC-001: Plugin loads via `claude --plugin-dir ./ray-ecosystem` ✓/✗
- SC-002: Both agents trigger on domain-appropriate queries ✓/✗
- SC-003: All 10 skills trigger on documented trigger phrases ✓/✗
- SC-004: Agent context under 60 lines each (checked in Step 3) ✓/✗
- SC-005: `plugin-validator` passes with no errors (checked in Step 1) ✓/✗
- SC-006: Skill descriptions distinguishable (checked in Step 2) ✓/✗

- [ ] **Step 6: Fix any issues found, then commit**

```bash
git add ray-ecosystem/
git commit -m "fix: address plugin validation feedback"
```

---

## Dependencies & Execution Order

- **Task 1** (scaffold): no dependencies — start here
- **Tasks 2-3** (agents): depend on Task 1. Can run in parallel with each other.
- **Tasks 4-13** (skills): depend on Task 1. All 10 can run in parallel.
  Each skill uses context7 to fetch current docs and expands source agent content.
- **Task 14** (README): depends on Tasks 2-13 (needs final component list).
- **Task 15** (validation): depends on all previous tasks.

### Parallel Opportunities

Tasks marked `[P]` (4-13) are fully independent — different directories,
no shared files. All 10 skills can be authored simultaneously.

Tasks 2 and 3 (agents) are also independent of each other.

Maximum parallelism: Tasks 2, 3, and 4-13 all run after Task 1 completes
(12 parallel streams).

```
Task 1 (scaffold)
    │
    ├── Task 2 (ray-expert agent) ──────────┐
    ├── Task 3 (rllib-expert agent) ─────────┤
    ├── Task 4 (ray-core-patterns) [P] ─────┤
    ├── Task 5 (ray-serve-deployment) [P] ──┤
    ├── Task 6 (ray-data-train) [P] ────────┤
    ├── Task 7 (ray-tune) [P] ──────────────┤
    ├── Task 8 (ray-cluster) [P] ───────────┤
    ├── Task 9 (rllib-env-setup) [P] ───────┤
    ├── Task 10 (rllib-training) [P] ───────┤
    ├── Task 11 (rllib-standards) [P] ──────┤
    ├── Task 12 (rllib-multi-agent) [P] ────┤
    └── Task 13 (rllib-advanced) [P] ───────┤
                                            │
                                      Task 14 (README)
                                            │
                                      Task 15 (validate)
```
