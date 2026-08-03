"""Tests for opt-in built-in plugins and plugin discovery."""
from datetime import datetime

from llm_platform.builtins.time import TimePlugin, get_current_time
from llm_platform.builtins.todo import TodoPlugin
from llm_platform.loader import (
    build_runtime,
    normalize_plugin_specs,
    resolve_plugin_module,
)


def test_normalize_plugin_specs_strings_and_mappings():
    specs = normalize_plugin_specs([
        "time",
        {"import": "llm_platform.builtins.todo"},
        {"name": "custom", "import": "myapp.plugins.custom"},
    ])
    assert specs == [
        {"name": "time"},
        {"name": "todo", "import": "llm_platform.builtins.todo"},
        {"name": "custom", "import": "myapp.plugins.custom"},
    ]


def test_resolve_plugin_module_builtins_and_app():
    assert resolve_plugin_module({"name": "time"}) == (
        "time",
        "llm_platform.builtins.time",
    )
    assert resolve_plugin_module({"name": "todo"}) == (
        "todo",
        "llm_platform.builtins.todo",
    )
    assert resolve_plugin_module({"name": "slack_search"}) == (
        "slack_search",
        "plugins.slack_search",
    )
    assert resolve_plugin_module({
        "name": "time",
        "import": "llm_platform.builtins.time",
    }) == ("time", "llm_platform.builtins.time")


def test_get_current_time_utc_iso():
    stamp = get_current_time("UTC")
    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None


def test_get_current_time_unknown_timezone():
    result = get_current_time("Not/AZone")
    assert "Unknown timezone" in result


def test_todo_session_scoped_operations():
    plugin = TodoPlugin()
    added = plugin.todo_add("write tests")
    assert added == {"id": 1, "text": "write tests", "done": False}
    assert plugin.todo_list() == [added]
    completed = plugin.todo_complete(1)
    assert completed["done"] is True
    assert plugin.todo_complete(99)["error"]
    assert plugin.todo_clear() == {"cleared": 1}
    assert plugin.todo_list() == []


def test_todo_instances_do_not_share_state():
    a = TodoPlugin()
    b = TodoPlugin()
    a.todo_add("only in a")
    assert b.todo_list() == []


def test_build_runtime_loads_builtins_when_listed():
    runtime = build_runtime({
        "model": "test/model",
        "plugins": ["time", "todo"],
        "max_tool_call_depth": 2,
        "compactor": {"kind": "null"},
        "api": {"kind": "completions"},
    })
    tools = runtime.tool_executor.tools
    assert "get_current_time" in tools
    assert "todo_add" in tools
    assert "todo_list" in tools
    assert "todo_complete" in tools
    assert "todo_clear" in tools
    prompt = runtime.session.system_prompt
    assert "## Todo" in prompt
    assert tools["todo_add"]("x")["id"] == 1


def test_build_runtime_without_plugins_has_no_builtin_tools():
    runtime = build_runtime({
        "model": "test/model",
        "plugins": [],
        "max_tool_call_depth": 1,
        "compactor": {"kind": "null"},
        "api": {"kind": "completions"},
    })
    assert runtime.tool_executor.tools == {}


def test_time_plugin_has_empty_prompt_section():
    assert TimePlugin().system_prompt_section() == ""
