# HOWTO: Build a simple math app

This procedure shows how to use `llm-platform` as a local package. The sample app asks for a math expression and sends it to the LLM. It uses no plugins.

Package version: **0.2.0**. See [CHANGELOG.md](CHANGELOG.md) for what is supported vs experimental.

## Prerequisites

- Python 3.10 or later
- An API key for the model provider (for example OpenAI)

## 1. Create the app directory

```bash
mkdir -p ~/Hack/math-llm && cd ~/Hack/math-llm
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install llm-platform

Install the package in editable mode from your local checkout:

```bash
pip install -e /Users/rdegraci/Hack/llm-platform
```

Editable mode links the source in place. Edits under `llm-platform/src/` apply without a reinstall.

## 3. Set the API key

Export a key in the shell:

```bash
export OPENAI_API_KEY=sk-...
```

You can also use `LITELLM_API_KEY`. Or put the key in a `.env` file in the app directory. The package loads dotenv automatically.

## 4. Add the config file

Create `llm_config.yaml`:

```yaml
model: openai/gpt-4.1-2025-04-14
plugins: []
max_tool_call_depth: 1
compactor:
  kind: null
api:
  kind: completions
completion_timeout: 60
tool_timeout: 30
log_events: false
tools:
  allow: null
  deny: []
```

Empty `plugins` is valid. This app only asks the model a question. It does not call tools.

Notes:

- `api.kind: completions` is the default and supported transport. You may omit the `api` block. To try the experimental Responses adapter for this app only, set `api.kind: responses` (function tools only; see the README).
- `completion_timeout` / `tool_timeout` are seconds. Use `null` to disable. A hung completion raises `TimeoutError`.
- Set `log_events: true` to log turn, completion, and tool events at INFO.
- `tools.allow` / `tools.deny` matter when you load plugins. With no plugins they have no effect.
- Invalid config (empty model, bad `api.kind`, and similar) raises `ValueError` from `load_config` / `build_runtime`.

Optional builtins (still off here): add `time` and/or `todo` under `plugins:` when you want them. See the README.

## 5. Add the app entry point

Create `main.py`:

```python
from llm_platform import configure_litellm_api_key, load_config, build_runtime

configure_litellm_api_key()
runtime = build_runtime(load_config())

expr = input("Math expression: ").strip()
if not expr:
    raise SystemExit("No expression given.")

answer = runtime.query(
    f"Compute this math expression and give only the result.\n\n{expr}"
)
print(answer)
```

To capture structured events in the app instead of (or as well as) logs:

```python
def on_event(event, payload):
    print(event, payload)

runtime = build_runtime(load_config(), on_event=on_event)
```

## 6. Run the app

```bash
python main.py
```

Example session:

```
Math expression: 2+2
4
```

## Optional next steps

- Loop with `while True` and call `runtime.reset()` between questions for a fresh session each time.
- Add a plugin later if you want the model to call a `calculate` tool instead of doing the arithmetic itself.
- Enable builtins with `plugins: [time, todo]` for clock and session checklist tools.

See the main [README](README.md) for plugin layout, transport, and the public API.
