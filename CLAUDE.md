# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Claude Code plugin marketplace and plugin collection** repository. It contains distributable Claude Code plugins (starting with the Ray ecosystem) and a marketplace index for plugin discovery and installation.

The goal is to build production-quality plugins that can be submitted to `anthropics/claude-plugins-official` or community marketplaces, and eventually host our own marketplace repo.

## Plugin Structure Convention

Each plugin lives in its own directory and follows the standard Claude Code plugin layout:

```
plugin-name/
├── .claude-plugin/
│   └── plugin.json       # Manifest: name, description, version, author
├── commands/              # Slash commands (SKILL.md files)
├── agents/                # Specialized subagents
├── skills/                # Auto-invoked or on-demand skills
├── hooks/                 # Lifecycle event handlers (PreToolUse, PostToolUse, etc.)
├── .mcp.json              # MCP server configuration (external tools)
└── README.md
```

**Component roles:**
- **CLAUDE.md** = always-on context (rules that apply to nearly every task)
- **Skills** = on-demand context, invoked by conversation matching or `/plugin-name:skill-name`
- **Agents** = isolated subagent contexts with specialized system prompts
- **Hooks** = fire on lifecycle events (pre/post tool use, session start/end, etc.)

## Marketplace Structure

The marketplace index is a GitHub repo that catalogs available plugins. Users interact via:
```
/plugin marketplace add <github-user>/<marketplace-repo>
/plugin install <plugin-name>@<directory>
```

## Development & Testing

Test plugins locally during development:
```bash
claude --plugin-dir ./plugin-name
```

Reload after changes without restarting:
```
/reload-plugins
```

## Speckit Integration

This repo uses speckit (`.specify/` directory) for feature specification and planning. Available commands:
- `/speckit.specify` — create/update feature spec
- `/speckit.plan` — generate implementation plan
- `/speckit.tasks` — generate task list from plan
- `/speckit.implement` — execute implementation
- `/speckit.clarify` — ask clarification questions about spec
- `/speckit.analyze` — cross-artifact consistency check
- `/speckit.constitution` — create/update project constitution

The constitution at `.specify/memory/constitution.md` defines the 5 core principles governing all plugin development. Review it before starting work.

## Current Scope

- **Ray ecosystem plugins**: Ray Core, Ray Serve, Ray Data, Ray Train, Ray Tune, RLlib
- License: MIT (abelyaev-ai)
