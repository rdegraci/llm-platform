"""Capability-agnostic LLM tool-calling runtime with a plugin host."""

from llm_platform.config import (
    apply_tool_filters,
    build_compactor,
    load_config,
    logging_event_handler,
    validate_config,
)
from llm_platform.loader import (
    CompositePlugin,
    build_runtime,
    completion_fn_for_config,
    normalize_plugin_specs,
)
from llm_platform.plugin_api import (
    Plugin,
    available_plugins,
    get_plugin,
    register_plugin,
)
from llm_platform.responses_adapter import responses_completion
from llm_platform.runtime import (
    Compactor,
    ConversationSession,
    DictToolExecutor,
    EventCallback,
    LLMRuntime,
    NullCompactor,
    SlidingWindowCompactor,
    configure_litellm_api_key,
    run_with_timeout,
)

__all__ = [
    "Compactor",
    "CompositePlugin",
    "ConversationSession",
    "DictToolExecutor",
    "EventCallback",
    "LLMRuntime",
    "NullCompactor",
    "Plugin",
    "SlidingWindowCompactor",
    "apply_tool_filters",
    "available_plugins",
    "build_compactor",
    "build_runtime",
    "completion_fn_for_config",
    "configure_litellm_api_key",
    "get_plugin",
    "load_config",
    "logging_event_handler",
    "normalize_plugin_specs",
    "register_plugin",
    "responses_completion",
    "run_with_timeout",
    "validate_config",
]

__version__ = "0.2.0"
