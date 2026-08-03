"""Smoke tests for llm-platform core types."""
from llm_platform import (
    CompositePlugin,
    Plugin,
    SlidingWindowCompactor,
    register_plugin,
)
from llm_platform.loader import _routing_preamble
from llm_platform.plugin_api import available_plugins


def test_sliding_window_preserves_system_and_last_turns():
    messages = [{"role": "system", "content": "sys"}]
    for i in range(5):
        messages.append({"role": "user", "content": f"u{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})

    compacted = SlidingWindowCompactor(max_turns=2).compact(messages)
    assert compacted[0] == {"role": "system", "content": "sys"}
    assert compacted[1]["content"] == "u3"
    assert len([m for m in compacted if m["role"] == "user"]) == 2


def test_register_plugin_requires_summary():
    try:

        @register_plugin
        class BadPlugin(Plugin):
            name = "bad_plugin"

            def tool_schemas(self):
                return []

            def tool_functions(self):
                return {}

        assert False, "expected ValueError"
    except ValueError as e:
        assert "summary" in str(e)


def test_routing_preamble_empty_for_single_plugin():
    @register_plugin
    class OnlyPlugin(Plugin):
        name = "only_plugin"
        summary = "Does one thing."

        def tool_schemas(self):
            return []

        def tool_functions(self):
            return {}

        def system_prompt_section(self, user_context=None):
            return "## Only\n\nTools here."

    plugin = OnlyPlugin()
    assert _routing_preamble([plugin]) == ""
    prompt = CompositePlugin([plugin]).system_prompt()
    assert "## Capabilities" not in prompt
    assert "## Only" in prompt
    assert "only_plugin" in available_plugins()
