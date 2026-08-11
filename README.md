# llm-platform

> Capability-agnostic LLM tool-calling runtime with a plugin host.

[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](#)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](#requirements)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`llm-platform` is a generic solver loop for apps that plug in tools like Slack, Jira, calc, and similar integrations. The package owns completion, session, compaction, and plugin composition. Apps own capabilities and product UI.

> Extracted from the host layer of `slack-search`.

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Plugins](#plugins)
- [Sample apps](#sample-apps)
- [Project layout](#project-layout)
- [Development](#development)
- [Status](#status)
- [Documentation](#documentation)
- [License](#license)

## Overview

Use `llm-platform` when you want a reusable model-driven tool loop without hardcoding domain logic into the host. Define capabilities as plugins, load them per app, and reuse the same runtime across multiple products.

## Features

- Tool-calling loop with injected `completion_fn` (default: LiteLLM Completions)
- Plugin host for schemas, callables, and optional prompt sections
- YAML config with validation (`validate_config`)
- Opt-in builtins: `time`, `todo` (session-scoped)
- Timeouts for completions and tools
- Structured events (`on_event` or `log_events`)
- Tool allow and deny lists
- Sliding-window conversation compaction
- Experimental OpenAI Responses transport (`api.kind: responses`)

## Requirements

- Python 3.10+
- A model API key, such as `OPENAI_API_KEY` or `LITELLM_API_KEY`

## Installation

Local editable install while developing:

```bash
pip install -e /path/to/llm-platform
```

From an app next to this repo:

```bash
pip install -e ../llm-platform
# or, from a sample app:
pip install -r requirements.txt   # uses -e ../..
```

## Quick start

```python
from llm_platform import configure_litellm_api_key, load_config, build_runtime

configure_litellm_api_key()
runtime = build_runtime(load_config())  # ./llm_config.yaml or LLM_CONFIG_PATH
print(runtime.query("Hello"))
```

Minimal `llm_config.yaml`:

```yaml
model: openai/gpt-4.1-2025-04-14
plugins: []
max_tool_call_depth: 3
api:
  kind: completions
```

## How it works

`llm-platform` sends the system prompt, tool schemas, and session messages to the model. If the model returns tool calls, the host runs them and continues until a normal answer or the depth limit.

The host does not choose tools in code. The model chooses tools from schemas and the prompt.

| Piece | Role |
| --- | --- |
| **Host** | Prompt assembly, model call, tool execution, session, compaction, limits |
| **Plugin** | One problem class: tools plus optional prompt section |
| **App** | Which plugins to load, plus config |

Write a plugin per domain. Load one or more plugins in an app. Reuse the same solver across apps.

## Configuration

| Key | Default | Description |
| --- | --- | --- |
| `model` | `openai/gpt-4.1-2025-04-14` | LiteLLM model id |
| `plugins` | `[]` | Plugin names and/or `{name, import}` mappings |
| `max_tool_call_depth` | `3` | Max tool rounds per user turn |
| `compactor.kind` | `sliding_window` | `sliding_window`, `null`, or `none` |
| `api.kind` | `completions` | `completions` (supported) or `responses` (experimental) |
| `completion_timeout` | `null` | Seconds; `null` disables |
| `tool_timeout` | `null` | Seconds; `null` disables |
| `log_events` | `false` | Log runtime events at INFO |
| `tools.allow` | `null` | Tool name allowlist (`null` = all) |
| `tools.deny` | `[]` | Tool names to remove after load |

Example with operational settings:

```yaml
model: openai/gpt-4.1-2025-04-14
plugins:
  - time
  - todo
  - my_capability
max_tool_call_depth: 3
compactor:
  kind: sliding_window
  max_turns: 20
api:
  kind: completions
completion_timeout: 60
tool_timeout: 30
log_events: true
tools:
  allow: null
  deny:
    - todo_clear
```

**Events:** pass `on_event(event, payload)` to `build_runtime`, or set `log_events: true`. Event names include `turn.start`, `turn.end`, `completion.start`, `completion.end`, `tool.call`, `tool.result`, and `error`.

**Timeouts:** the runtime uses a worker thread. On timeout, completion raises; a tool returns an error string and the loop continues. The worker may still finish in the background.

## Plugins

1. Create `plugins/<name>/` on the app import path.
2. Subclass `Plugin`, set `name` and `summary`, implement `tool_schemas()` and `tool_functions()`.
3. Decorate with `@register_plugin`.
4. List the name under `plugins:` in config.

`system_prompt_section()` is optional. Keep each plugin focused on one domain.

Reserved builtin names: `time`, `todo` (see below). Other names import as `plugins.<name>`.

Explicit module path:

```yaml
plugins:
  - name: jira
    import: myapp.capabilities.jira
```

### Builtins (opt-in)

| Name | Tools | Notes |
| --- | --- | --- |
| `time` | `get_current_time` | ISO-8601 + IANA timezone |
| `todo` | `todo_add`, `todo_list`, `todo_complete`, `todo_clear` | Session-scoped only |

Off by default. Domain APIs like Slack and Jira stay in the app, not in builtins.

### Experimental: Responses API

```yaml
api:
  kind: responses
```

Maps Completions-shaped sessions through `litellm.responses`. Function tools only. Completions remains the supported default.

## Sample apps

| Path | Description |
| --- | --- |
| [`sample_apps/simple/`](sample_apps/simple/) | Math Q&A, no plugins |
| [`sample_apps/complex/`](sample_apps/complex/) | App-local `calculate` plugin |

```bash
cd sample_apps/simple   # or sample_apps/complex
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python main.py
```

## Project layout

```text
llm-platform/
├── src/llm_platform/
│   ├── runtime.py           # Tool loop, session, timeouts, events
│   ├── plugin_api.py        # Plugin ABC + registry
│   ├── loader.py            # build_runtime, plugin discovery
│   ├── config.py            # YAML load, validation, tool filters
│   ├── responses_adapter.py # Experimental Responses transport
│   └── builtins/            # Opt-in time + todo plugins
├── sample_apps/
│   ├── simple/
│   └── complex/
├── tests/
├── HOWTO.md
└── CHANGELOG.md
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Status

Suitable for local multi-project use via editable install. Not a multi-tenant production platform.

| Surface | Status |
| --- | --- |
| Completions transport | Supported |
| Plugin host + tool loop | Supported |
| Builtins `time` / `todo` | Supported (opt-in) |
| Responses adapter | Experimental |
| Timeouts | Supported (thread-based limits) |

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Documentation

| Doc | Description |
| --- | --- |
| [HOWTO.md](HOWTO.md) | Step-by-step math app from scratch |
| [CHANGELOG.md](CHANGELOG.md) | Version history and stability notes |
| [sample_apps/simple/](sample_apps/simple/) | Minimal runnable sample |
| [sample_apps/complex/](sample_apps/complex/) | Plugin sample |

## License

Copyright 2026 Rodney Degracia

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
