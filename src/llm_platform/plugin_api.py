"""Plugin contract and registry for the LLM framework.

A *plugin* bundles one capability domain (Slack, Jira, GitHub, ...) into three
things the generic runtime needs: tool schemas to advertise to the model, the
callables that implement those tools, and a system-prompt section describing how
to use them. Plugins are capability layers; the runtime in ``runtime`` is the
capability-agnostic host that drives them.

Plugins self-register by decorating their class with :func:`register_plugin`.
The loader imports the package for each configured plugin name, which triggers
registration, then instantiates it via :func:`get_plugin`.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type

import logging

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """A single capability domain wired into the runtime.

    Subclasses set two required class attributes:

    - ``name`` — unique registry key; also the ``plugins/<name>`` package dir.
    - ``summary`` — one-line, user-facing capability description. The loader
      uses this in a routing preamble when more than one plugin is loaded, so
      the model can choose between them. Keep it short and verb-led, e.g.
      "Search and read messages from your Slack workspace."

    ``system_prompt_section`` is optional and defaults to empty.
    """

    name: str = ""
    summary: str = ""

    @abstractmethod
    def tool_schemas(self) -> List[Dict[str, Any]]:
        """Return the JSON tool schemas advertised to the model."""

    @abstractmethod
    def tool_functions(self) -> Dict[str, Callable[..., Any]]:
        """Return a mapping of tool name -> callable implementing it."""

    def system_prompt_section(self, user_context: Any = None) -> str:
        """Return capability-specific prompt text concatenated by the loader.

        Contract — the returned section MUST contain:

        - A capability heading (``## <Capability name>``).
        - Tool descriptions and when to use each tool.
        - Workflow specific to this capability.
        - Output formatting rules *only where they diverge* from the host base
          prompt (e.g. Slack message bodies as Markdown blockquotes).

        It MUST NOT contain:

        - An assistant role/identity declaration ("You are a ... assistant") —
          the host base prompt owns the role exactly once.
        - Today's date — the host already injects it.
        - Generic style or concision guidance — owned by the host.
        - Comparisons to, or routing between, other plugins — that is the
          host's job via ``summary`` and the routing preamble; plugin sections
          must not know about each other.

        Default implementation returns an empty string (plugin contributes
        tools only, no prompt section).
        """
        return ""


_REGISTRY: Dict[str, Type[Plugin]] = {}


def register_plugin(cls: Type[Plugin]) -> Type[Plugin]:
    """Class decorator that registers a plugin under its ``name``.

    Raises ValueError on a missing/empty name or a duplicate registration.
    """
    name = getattr(cls, "name", "")
    if not name:
        raise ValueError(
            f"Plugin {cls.__name__!r} must define a non-empty 'name' class attribute."
        )
    summary = getattr(cls, "summary", "")
    if not summary:
        raise ValueError(
            f"Plugin {cls.__name__!r} must define a non-empty 'summary' class "
            "attribute — a one-line capability description used for multi-plugin "
            "routing."
        )
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Plugin name {name!r} is already registered to {existing.__name__!r}."
        )
    _REGISTRY[name] = cls
    logger.debug("Registered plugin %r -> %s", name, cls.__name__)
    return cls


def get_plugin(name: str) -> Plugin:
    """Instantiate the plugin registered under ``name``.

    Raises KeyError (with the list of available names) if it is not registered.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"No plugin registered under {name!r}. Available: {available_plugins()}"
        )
    return cls()


def available_plugins() -> List[str]:
    """Return the sorted names of all registered plugins."""
    return sorted(_REGISTRY)
