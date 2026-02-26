---
name: IDE Agent Kit
version: 1.1.1
description: A local-first development sidekick that securely bridges your LLM to your IDE, terminal, and browser without external servers.
author: ThinkOff
tags: [developer-tools, local-first, ide, automation]
---

# IDE Agent Kit

A powerful, local-first development sidekick that securely bridges your LLM to your IDE, terminal, and browser.

**Local-first. No external server by default.** All interactions happen directly on your machine, ensuring maximum privacy and security for your codebase.

## Security Model
- **Local Execution:** Commands and scripts run locally on your machine.
- **Explicit Consent:** Destructive actions require explicit user approval.
- **No External Telemetry:** We do not send your code or environment data to external servers.

## Network Behavior
| Endpoint | Purpose | Required |
| --- | --- | --- |
| `localhost:*` | Local application testing and IDE communication | Yes |
| LLM Provider URLs | API calls to your configured LLM (e.g. Anthropic, OpenAI) | Yes |

## The Four Modes
1. **Planning Mode:** Research the codebase, understand requirements, and design approaches.
2. **Execution Mode:** Write code, make changes, and implement designs.
3. **Verification Mode:** Test changes, run verification steps, and validate correctness.
4. **Agentic Mode:** Autonomous operation with structured artifact generation and task boundaries.

## Quick Start
1. Install the MCP server globally: `npm install -g ide-agent-kit`
2. Start the agent in your project directory: `ide-agent`
3. The agent will read your workspace context and await your instructions.
