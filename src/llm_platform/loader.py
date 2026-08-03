"""Framework loader: compose configured plugins into a single LLMRuntime.

Given a config dict (see ``llm_platform.config``), this imports each named
plugin package so it self-registers, instantiates the plugins, merges their
tool schemas / functions / prompt sections via :class:`CompositePlugin`, and
injects the result into the generic :class:`llm_platform.runtime.LLMRuntime`
host.

Plugin specs in config may be:

- a string name — opt-in builtins (``time``, ``todo``) resolve to
  ``llm_platform.builtins.<name>``; all other names import ``plugins.<name>``
- a mapping ``{name, import}`` — import an explicit module path (``name``
  defaults to the last segment of ``import`` when omitted)
"""
import importlib
import logging
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import litellm

from llm_platform.config import apply_tool_filters, build_compactor, logging_event_handler, validate_config
from llm_platform.plugin_api import Plugin, get_plugin
from llm_platform.responses_adapter import responses_completion
from llm_platform.runtime import DictToolExecutor, EventCallback, LLMRuntime

logger = logging.getLogger(__name__)

# Opt-in built-in plugins shipped with the package. Not loaded unless listed
# under config ``plugins``.
BUILTIN_PLUGIN_MODULES: Dict[str, str] = {
    "time": "llm_platform.builtins.time",
    "todo": "llm_platform.builtins.todo",
}

PluginSpec = Dict[str, str]


def completion_fn_for_config(config: Dict[str, Any]) -> Callable[..., Any]:
    """Select the transport callable for ``build_runtime``.

    Reads ``config["api"]["kind"]``:

    - ``completions`` (default) — ``litellm.completion``
    - ``responses`` — experimental Responses adapter (function tools only)

    Apps that construct :class:`~llm_platform.runtime.LLMRuntime` by hand can
    still pass ``completion_fn`` explicitly and skip this helper.
    """
    api = config.get("api") or {}
    kind = api.get("kind", "completions")
    if kind == "completions":
        return litellm.completion
    if kind == "responses":
        logger.info(
            "Using experimental Responses API adapter (api.kind=responses). "
            "Completions remains the default supported transport."
        )
        return responses_completion
    raise ValueError(
        f"Unknown api.kind: {kind!r}. Expected 'completions' or 'responses'."
    )


def normalize_plugin_specs(
    plugins: Optional[List[Union[str, Dict[str, Any]]]],
) -> List[PluginSpec]:
    """Normalize config plugin entries to ``{name[, import]}`` dicts."""
    specs: List[PluginSpec] = []
    for entry in plugins or []:
        if isinstance(entry, str):
            if not entry.strip():
                raise ValueError("Plugin name must be a non-empty string.")
            specs.append({"name": entry.strip()})
            continue
        if isinstance(entry, dict):
            module = entry.get("import")
            name = entry.get("name")
            if module and not name:
                name = str(module).rsplit(".", 1)[-1]
            if not name:
                raise ValueError(
                    f"Plugin entry requires 'name' or 'import', got {entry!r}."
                )
            spec: PluginSpec = {"name": str(name)}
            if module:
                spec["import"] = str(module)
            specs.append(spec)
            continue
        raise ValueError(
            f"Invalid plugin entry {entry!r}; expected a string or "
            "{name, import} mapping."
        )
    return specs


def resolve_plugin_module(spec: PluginSpec) -> Tuple[str, str]:
    """Return ``(registry_name, module_path)`` for a normalized plugin spec."""
    name = spec["name"]
    if "import" in spec:
        return name, spec["import"]
    if name in BUILTIN_PLUGIN_MODULES:
        return name, BUILTIN_PLUGIN_MODULES[name]
    return name, f"plugins.{name}"


def BASE_SYSTEM_PROMPT(user_context: Any = None) -> str:
    """Capability-agnostic host preamble shared by every plugin.

    Plugin-specific tool descriptions, workflows, and formatting are appended by
    each plugin's ``system_prompt_section``.
    """
    today = date.today().isoformat()
    return f"""You are a helpful assistant with access to a set of tools, described
below. Use them to find and retrieve information the user asks for, then answer
clearly. Today's date is {today}.

## Tool use

- Call tools when they help answer the request; don't call them speculatively.
- You have a budget of a few tool rounds per user turn — plan accordingly.
- Never fabricate tool results. If a tool fails or returns nothing, say so.

## Output (rendered as Markdown in the terminal)

- Use a numbered list when presenting multiple results.
- Use inline code (backticks) for short identifiers, IDs, or technical terms.
- Format links as Markdown links: [label](https://...).

## Style

- Be concise. Quote selectively rather than paraphrasing the obvious.
- For long content, summarize and link rather than pasting it in full.
- Ask clarifying questions only when the request is genuinely ambiguous — don't
  ask for permission to do reasonable things.
"""


def _routing_preamble(plugins: List[Plugin]) -> str:
    """Markdown block telling the model how to choose between capabilities.

    Returns an empty string when ``len(plugins) <= 1`` so single-plugin prompts
    pay zero tokens for routing they don't need. With multiple plugins, lists
    each plugin's ``name`` and ``summary`` as a bullet under ``## Capabilities``.
    """
    if len(plugins) <= 1:
        return ""
    lines = [
        "## Capabilities",
        "",
        "You have access to the following capabilities. For each user request, "
        "pick the capability whose tools fit best; you may use more than one "
        "across a turn if the request spans them.",
    ]
    for p in plugins:
        lines.append(f"- **{p.name}** — {p.summary}")
    return "\n".join(lines)


class CompositePlugin:
    """Merge several plugins into one tool/prompt surface for the runtime."""

    def __init__(self, plugins: List[Plugin]):
        self.plugins = list(plugins)

    def tool_schemas(self) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: Dict[str, str] = {}
        for p in self.plugins:
            for schema in p.tool_schemas():
                name = schema.get("function", {}).get("name")
                if name in seen:
                    raise ValueError(
                        f"Tool name collision: {name!r} declared by both "
                        f"{seen[name]!r} and {p.name!r}."
                    )
                seen[name] = p.name
                merged.append(schema)
        return merged

    def tool_functions(self) -> Dict[str, Callable[..., Any]]:
        merged: Dict[str, Callable[..., Any]] = {}
        owner: Dict[str, str] = {}
        for p in self.plugins:
            for name, fn in p.tool_functions().items():
                if name in merged:
                    raise ValueError(
                        f"Tool function collision: {name!r} provided by both "
                        f"{owner[name]!r} and {p.name!r}."
                    )
                merged[name] = fn
                owner[name] = p.name
        return merged

    def system_prompt(self, user_context: Any = None) -> str:
        parts = [BASE_SYSTEM_PROMPT(user_context)]
        preamble = _routing_preamble(self.plugins)
        if preamble:
            parts.append(preamble)
        for p in self.plugins:
            section = p.system_prompt_section(user_context)
            if section:
                parts.append(section)
        return "\n\n".join(parts)


def _ensure_imported(spec: PluginSpec) -> str:
    """Import the plugin module so ``@register_plugin`` runs. Return registry name."""
    name, module_path = resolve_plugin_module(spec)
    try:
        importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if module_path.startswith("plugins."):
            raise ModuleNotFoundError(
                f"Could not import plugin package {module_path!r} for configured "
                f"plugin {name!r}. Expected a package at plugins/{name}/."
            ) from e
        raise ModuleNotFoundError(
            f"Could not import plugin module {module_path!r} for configured "
            f"plugin {name!r}."
        ) from e
    return name


def build_runtime(
    config: Dict[str, Any],
    on_event: Optional[EventCallback] = None,
) -> LLMRuntime:
    """Build an :class:`LLMRuntime` from a config dict by composing its plugins.

    Pass ``on_event`` for a structured event callback, or set ``log_events: true``
    in config to log events via the standard logger.
    """
    config = validate_config(config)
    specs = normalize_plugin_specs(config.get("plugins") or [])
    if not specs:
        logger.warning("No plugins configured; runtime will have no tools.")

    names = [_ensure_imported(spec) for spec in specs]
    plugins = [get_plugin(name) for name in names]
    composite = CompositePlugin(plugins)

    schemas = composite.tool_schemas()
    functions = composite.tool_functions()
    tools_cfg = config.get("tools") or {}
    schemas, functions = apply_tool_filters(
        schemas,
        functions,
        allow=tools_cfg.get("allow"),
        deny=tools_cfg.get("deny"),
    )

    event_cb = on_event
    if event_cb is None and config.get("log_events"):
        event_cb = logging_event_handler

    return LLMRuntime(
        model=config["model"],
        tools=schemas,
        tool_executor=DictToolExecutor(functions),
        system_prompt_factory=composite.system_prompt,
        max_tool_call_depth=config["max_tool_call_depth"],
        completion_fn=completion_fn_for_config(config),
        compactor=build_compactor(config),
        completion_timeout=config.get("completion_timeout"),
        tool_timeout=config.get("tool_timeout"),
        on_event=event_cb,
    )
