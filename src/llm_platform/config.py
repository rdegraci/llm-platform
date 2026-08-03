"""Config loader for the LLM runtime.

Loads runtime knobs (model, tool-call depth, compactor, log level) from a
YAML file into a plain dict. The library itself takes a dict; this loader
is the optional convenience for apps that prefer YAML.
"""
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_platform.runtime import Compactor, NullCompactor, SlidingWindowCompactor

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "llm_config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "model": "openai/gpt-4.1-2025-04-14",
    "plugins": [],
    "max_tool_call_depth": 3,
    "compactor": {
        "kind": "sliding_window",
        "max_turns": 20,
    },
    # Transport selector. "completions" is the default and supported path.
    # "responses" is an experimental OpenAI-oriented opt-in (see responses_adapter).
    "api": {
        "kind": "completions",
    },
    # Seconds; null disables. Worker threads are not killed on timeout.
    "completion_timeout": None,
    "tool_timeout": None,
    # Tool name filters applied after plugins load. allow=null means all tools.
    "tools": {
        "allow": None,
        "deny": [],
    },
    # When true, build_runtime attaches a logging on_event handler.
    "log_events": False,
    # None means "do not override whatever LOG_LEVEL / basicConfig set up".
    "log_level": None,
}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config, merging onto defaults, then validate.

    Search order:
      1. explicit ``path`` argument
      2. ``LLM_CONFIG_PATH`` env var
      3. ``./llm_config.yaml``

    A missing file is not an error — defaults are returned (with a warning).
    Invalid values raise ``ValueError``.
    """
    if path is None:
        path = os.environ.get("LLM_CONFIG_PATH", DEFAULT_CONFIG_PATH)

    config: Dict[str, Any] = {
        "model": DEFAULT_CONFIG["model"],
        "plugins": list(DEFAULT_CONFIG["plugins"]),
        "max_tool_call_depth": DEFAULT_CONFIG["max_tool_call_depth"],
        "compactor": dict(DEFAULT_CONFIG["compactor"]),
        "api": dict(DEFAULT_CONFIG["api"]),
        "completion_timeout": DEFAULT_CONFIG["completion_timeout"],
        "tool_timeout": DEFAULT_CONFIG["tool_timeout"],
        "tools": {
            "allow": DEFAULT_CONFIG["tools"]["allow"],
            "deny": list(DEFAULT_CONFIG["tools"]["deny"]),
        },
        "log_events": DEFAULT_CONFIG["log_events"],
        "log_level": DEFAULT_CONFIG["log_level"],
    }

    p = Path(path)
    if not p.exists():
        logger.warning("Config file %s not found; using built-in defaults.", p)
        return validate_config(config)

    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "PyYAML is required to load llm_config.yaml. Install with `pip install pyyaml`."
        ) from e

    with p.open() as f:
        loaded = yaml.safe_load(f) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {p} must contain a YAML mapping at the top level.")

    if "model" in loaded:
        config["model"] = loaded["model"]
    if "plugins" in loaded:
        plugins = loaded["plugins"]
        if not isinstance(plugins, list) or not _plugins_entries_valid(plugins):
            raise ValueError(
                "config 'plugins' must be a list of non-empty strings or "
                "{name, import} mappings."
            )
        config["plugins"] = plugins
    if "max_tool_call_depth" in loaded:
        config["max_tool_call_depth"] = loaded["max_tool_call_depth"]
    if "compactor" in loaded and isinstance(loaded["compactor"], dict):
        config["compactor"].update(loaded["compactor"])
    if "api" in loaded and isinstance(loaded["api"], dict):
        config["api"].update(loaded["api"])
    if "completion_timeout" in loaded:
        config["completion_timeout"] = loaded["completion_timeout"]
    if "tool_timeout" in loaded:
        config["tool_timeout"] = loaded["tool_timeout"]
    if "tools" in loaded and isinstance(loaded["tools"], dict):
        if "allow" in loaded["tools"]:
            config["tools"]["allow"] = loaded["tools"]["allow"]
        if "deny" in loaded["tools"]:
            config["tools"]["deny"] = loaded["tools"]["deny"]
    if "log_events" in loaded:
        config["log_events"] = loaded["log_events"]
    if "log_level" in loaded:
        config["log_level"] = loaded["log_level"]

    return validate_config(config)


def _plugins_entries_valid(plugins: list) -> bool:
    """Accept string names or ``{name?, import?}`` mappings with at least one key."""
    for entry in plugins:
        if isinstance(entry, str):
            if not entry.strip():
                return False
            continue
        if isinstance(entry, dict):
            if "name" in entry or "import" in entry:
                continue
            return False
        return False
    return True


def _validate_timeout(name: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"config '{name}' must be a positive number of seconds, or null.")
    if value <= 0:
        raise ValueError(f"config '{name}' must be a positive number of seconds, or null.")
    return float(value)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a config dict. Raise ``ValueError`` on bad values. Return config."""
    from llm_platform.loader import normalize_plugin_specs

    model = config.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("config 'model' must be a non-empty string.")

    depth = config.get("max_tool_call_depth")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        raise ValueError("config 'max_tool_call_depth' must be a non-negative integer.")

    api = config.get("api") or {}
    if not isinstance(api, dict):
        raise ValueError("config 'api' must be a mapping.")
    kind = api.get("kind", "completions")
    if kind not in ("completions", "responses"):
        raise ValueError(
            f"Unknown api.kind: {kind!r}. Expected 'completions' or 'responses'."
        )

    try:
        normalize_plugin_specs(config.get("plugins") or [])
    except ValueError as e:
        raise ValueError(f"config 'plugins' is invalid: {e}") from e

    config["completion_timeout"] = _validate_timeout(
        "completion_timeout", config.get("completion_timeout")
    )
    config["tool_timeout"] = _validate_timeout(
        "tool_timeout", config.get("tool_timeout")
    )

    tools_cfg = config.get("tools")
    if tools_cfg is None:
        tools_cfg = {"allow": None, "deny": []}
        config["tools"] = tools_cfg
    if not isinstance(tools_cfg, dict):
        raise ValueError("config 'tools' must be a mapping with optional allow/deny.")
    allow = tools_cfg.get("allow")
    deny = tools_cfg.get("deny", [])
    if allow is not None:
        if not isinstance(allow, list) or not all(isinstance(x, str) and x for x in allow):
            raise ValueError("config 'tools.allow' must be null or a list of non-empty strings.")
    if deny is None:
        deny = []
        tools_cfg["deny"] = deny
    if not isinstance(deny, list) or not all(isinstance(x, str) and x for x in deny):
        raise ValueError("config 'tools.deny' must be a list of non-empty strings.")

    log_events = config.get("log_events", False)
    if not isinstance(log_events, bool):
        raise ValueError("config 'log_events' must be a boolean.")

    comp_cfg = config.get("compactor") or {}
    if not isinstance(comp_cfg, dict):
        raise ValueError("config 'compactor' must be a mapping.")
    # YAML ``kind: null`` loads as Python None — treat it as the null compactor.
    kind_c = comp_cfg.get("kind", "null")
    if kind_c is None:
        kind_c = "null"
        comp_cfg["kind"] = "null"
    if kind_c not in ("null", "none", "sliding_window"):
        raise ValueError(f"Unknown compactor kind: {kind_c!r}")

    return config


def apply_tool_filters(
    schemas: List[Dict[str, Any]],
    functions: Dict[str, Callable[..., Any]],
    allow: Optional[List[str]],
    deny: Optional[List[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    """Filter tool schemas and callables by allow/deny lists.

    ``allow=None`` means all tools. Deny is applied after allow. Unknown names
    in ``allow`` raise ``ValueError``.
    """
    deny_set = set(deny or [])
    if allow is not None:
        allow_set = set(allow)
        missing = sorted(allow_set - set(functions))
        if missing:
            raise ValueError(
                f"tools.allow names not found in loaded tools: {missing}. "
                f"Available: {sorted(functions)}"
            )
    else:
        allow_set = None

    def keep(name: Optional[str]) -> bool:
        if not name:
            return False
        if allow_set is not None and name not in allow_set:
            return False
        if name in deny_set:
            return False
        return True

    filtered_functions = {name: fn for name, fn in functions.items() if keep(name)}
    filtered_schemas: List[Dict[str, Any]] = []
    for schema in schemas:
        name = schema.get("function", {}).get("name")
        if keep(name):
            filtered_schemas.append(schema)
    return filtered_schemas, filtered_functions


def build_compactor(config: Dict[str, Any]) -> Compactor:
    """Construct a Compactor instance from a config dict."""
    comp_cfg = config.get("compactor") or {}
    kind = comp_cfg.get("kind", "null")
    if kind is None:
        kind = "null"
    if kind in ("null", "none"):
        return NullCompactor()
    if kind == "sliding_window":
        return SlidingWindowCompactor(max_turns=int(comp_cfg.get("max_turns", 20)))
    raise ValueError(f"Unknown compactor kind: {kind!r}")


def logging_event_handler(event: str, payload: Dict[str, Any]) -> None:
    """Default structured event sink used when ``log_events: true``."""
    logger.info("runtime.%s %s", event, payload)
