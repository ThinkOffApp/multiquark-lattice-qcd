---
name: IDE Agent Kit
description: Local-first agent orchestration for IDE, terminal, and browser workflows without a required external backend.
version: 1.1.2
metadata:
  openclaw:
    requires:
      bins:
        - node
        - npm
    homepage: https://github.com/ThinkOffApp/ide-agent-kit
    install:
      - kind: node
        package: ide-agent-kit
        bins:
          - ide-agent
---

# IDE Agent Kit

IDE Agent Kit runs on your machine and coordinates agent work across your IDE, terminal, and browser with a local-first architecture. Default operation does not require a network credential, and remote integrations are optional additions rather than mandatory runtime dependencies.

## Identity and Key Scope

This skill does not require `ANTIGRAVITY_API_KEY` for local orchestration. If you later connect ThinkOff network services, antfarm.world acts as the shared identity provider and the resulting key is user-scoped by default. A user-scoped key cannot perform admin or moderator actions unless elevated credentials are explicitly issued.

## Security and Runtime Model

Execution remains local, user initiated, and observable. The tool is designed for explicit command invocation rather than background autonomous shell execution, and destructive operations should stay behind normal approval controls in your agent runtime policy.

## Quick Start

```bash
npm install -g ide-agent-kit
ide-agent
```
