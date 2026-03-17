<!--
  Sync Impact Report
  ==================
  Version change: N/A (template) → 1.0.0
  Modified principles: N/A (initial ratification)
  Added sections:
    - Core Principles (5 principles)
    - Marketplace Readiness Requirements
    - Development Workflow
    - Governance
  Removed sections: None
  Templates requiring updates:
    - .specify/templates/plan-template.md — ✅ no changes needed (Constitution Check section is generic)
    - .specify/templates/spec-template.md — ✅ no changes needed
    - .specify/templates/tasks-template.md — ✅ no changes needed
    - CLAUDE.md — ✅ no changes needed (already aligned)
    - README.md — ⚠️ pending: expand with project principles summary once plugins exist
  Follow-up TODOs: None
-->

# AI Agentic Plugins Constitution

## Core Principles

### I. Context7-First Documentation

All library and framework documentation MUST be resolved through
the context7 MCP server (`resolve-library-id` → `query-docs`).
Do not rely on training data or guesswork for API surfaces,
configuration options, or version-specific behavior.

**Rationale**: Plugins target fast-moving ecosystems (Ray, etc.)
where APIs change between versions. Context7 provides up-to-date,
version-pinned documentation that prevents stale advice.

### II. Plugin-Dev Tooling

All plugin scaffolding, component creation, and structural
validation MUST use the `plugin-dev` plugin and its agents
(`plugin-validator`, `agent-creator`, `skill-reviewer`).

- Use `/plugin-dev:create-plugin` for new plugins
- Use `/plugin-dev:plugin-structure` for layout guidance
- Use `/plugin-dev:skill-development` for skills
- Use `/plugin-dev:agent-development` for agents
- Use `/plugin-dev:hook-development` for hooks
- Use `/plugin-dev:command-development` for commands
- Validate with `plugin-validator` before any commit

**Rationale**: Consistent structure across all plugins in the
collection is non-negotiable for marketplace acceptance.

### III. Clean Separation of Concerns

Each plugin MUST own a single, well-defined domain. Within a
plugin, every component (skill, agent, command, hook) MUST have
a description that is:

- **Laconic**: one sentence, no filler
- **Distinguishable**: a reader scanning the list can immediately
  tell which component to use without reading the body
- **Non-overlapping**: no two components in the same plugin
  should handle the same task

If two plugins share responsibility for a concept, refactor
until ownership is unambiguous.

**Rationale**: Claude Code selects skills and agents by matching
descriptions to user intent. Vague or overlapping descriptions
cause mis-routing and user frustration.

### IV. Marketplace Standards Compliance

Every plugin MUST be publishable to third-party marketplaces
(`anthropics/claude-plugins-official`, community directories)
at all times. This means:

- Valid `.claude-plugin/plugin.json` with name, description,
  version (semver), and author
- A `README.md` in each plugin root explaining purpose,
  installation, and usage
- No hardcoded paths, secrets, or environment-specific config
- Semantic versioning: MAJOR for breaking changes, MINOR for
  new features, PATCH for fixes
- Each plugin directory is self-contained — it MUST work when
  installed in isolation via `/plugin install`

**Rationale**: The goal is marketplace distribution. A plugin
that only works inside this monorepo is not a plugin.

### V. Minimal Viable Plugin

A plugin MUST include only the components it actually needs.
Do not scaffold empty `commands/`, `agents/`, `hooks/`, or
`skills/` directories preemptively. Add components when there
is a concrete use case.

- Start with the smallest useful surface (often one skill or
  one agent)
- Justify every additional component against Principle III
- Prefer a skill over an agent when no isolated context or
  autonomous execution is required

**Rationale**: Over-engineered plugins are harder to maintain,
review, and explain. Simpler plugins get adopted faster.

## Marketplace Readiness Requirements

Before a plugin is submitted to any marketplace:

1. `plugin-validator` agent MUST pass with no errors
2. All skill and agent descriptions MUST satisfy Principle III
3. Plugin MUST be testable via `claude --plugin-dir ./plugin-name`
4. README MUST include: purpose, installation, component list,
   and at least one usage example
5. Version in `plugin.json` MUST follow semver and match the
   latest git tag (when tagged)

## Development Workflow

1. **Spec** — use `/speckit.specify` to define what the plugin does
2. **Plan** — use `/speckit.plan` to design components and structure
3. **Scaffold** — use `plugin-dev` tooling (Principle II) to create
   the plugin skeleton
4. **Implement** — write skills, agents, hooks; resolve library docs
   via context7 (Principle I)
5. **Validate** — run `plugin-validator`; test with `--plugin-dir`
6. **Review** — verify clean separation (Principle III), marketplace
   compliance (Principle IV), and minimality (Principle V)

## Governance

This constitution supersedes ad-hoc conventions. All plugin
development in this repository MUST comply with these principles.

- Amendments require updating this file, bumping the version,
  and noting changes in the Sync Impact Report comment above
- CLAUDE.md provides runtime development guidance and MUST stay
  aligned with this constitution
- When a principle conflicts with a marketplace requirement,
  the marketplace requirement wins

**Version**: 1.0.0 | **Ratified**: 2026-03-17 | **Last Amended**: 2026-03-17
