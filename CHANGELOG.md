# Changelog

## 0.2.0 — 2026-08-03

### Added

- Opt-in builtins: `time`, `todo` (session-scoped checklist)
- Plugin specs: string names, builtins allowlist, `{name, import}` mappings
- Experimental Responses transport via `api.kind: responses`
- `completion_timeout` / `tool_timeout` (seconds; null disables)
- Structured runtime events via `on_event` or `log_events: true`
- Tool filters: `tools.allow` / `tools.deny`
- `validate_config()` — fail fast on bad model, api.kind, plugins, timeouts

### Stability

| Surface | Status |
|---------|--------|
| Completions transport (`api.kind: completions`) | Supported default |
| Plugin host, tool loop, sliding-window compactor | Supported |
| Builtins `time` / `todo` | Supported, opt-in |
| Responses adapter (`api.kind: responses`) | **Experimental** |
| Timeouts (thread-based; worker not killed) | Supported with known limits |

### Notes

- Invalid `plugins` in YAML now raises instead of silently falling back.
- Default behavior with an empty/missing config is unchanged: Completions, no plugins, no timeouts.

## 0.1.0

Initial extract: runtime, plugin API, loader, YAML config, sliding-window compactor.
