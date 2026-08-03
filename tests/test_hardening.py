"""Tests for timeouts, events, tool filters, and config validation."""
import time
from typing import Any, Dict, List, Tuple

import pytest

from llm_platform.config import apply_tool_filters, validate_config
from llm_platform.loader import build_runtime
from llm_platform.runtime import DictToolExecutor, LLMRuntime, run_with_timeout


def _assistant_text(content: str) -> Dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _assistant_tools(calls: List[Dict[str, str]]) -> Dict[str, Any]:
    tool_calls = [
        {
            "id": c["id"],
            "type": "function",
            "function": {"name": c["name"], "arguments": c.get("arguments", "{}")},
        }
        for c in calls
    ]
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            }
        }]
    }


class ScriptedCompletion:
    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_run_with_timeout_ok():
    assert run_with_timeout(lambda: 7, 1.0) == 7


def test_run_with_timeout_raises():
    with pytest.raises(TimeoutError):
        run_with_timeout(time.sleep, 0.05, 1.0)


def test_completion_timeout_emits_error_and_raises():
    events: List[Tuple[str, Dict[str, Any]]] = []

    def slow_completion(**kwargs):
        time.sleep(1.0)
        return _assistant_text("late")

    runtime = LLMRuntime(
        model="test/model",
        tools=[],
        tool_executor=DictToolExecutor({}),
        system_prompt_factory=lambda _ctx: "sys",
        completion_fn=slow_completion,
        completion_timeout=0.05,
        on_event=lambda e, p: events.append((e, p)),
    )
    with pytest.raises(TimeoutError):
        runtime.query("hi")
    assert any(e == "error" and p.get("phase") == "completion" for e, p in events)


def test_tool_timeout_becomes_tool_result_and_continues():
    events: List[Tuple[str, Dict[str, Any]]] = []
    script = ScriptedCompletion([
        _assistant_tools([{"id": "call_1", "name": "slow", "arguments": "{}"}]),
        _assistant_text("recovered"),
    ])

    def slow():
        time.sleep(1.0)
        return "done"

    runtime = LLMRuntime(
        model="test/model",
        tools=[{
            "type": "function",
            "function": {"name": "slow", "parameters": {"type": "object", "properties": {}}},
        }],
        tool_executor=DictToolExecutor({"slow": slow}),
        system_prompt_factory=lambda _ctx: "sys",
        completion_fn=script,
        tool_timeout=0.05,
        on_event=lambda e, p: events.append((e, p)),
    )
    assert runtime.query("go") == "recovered"
    tool_msgs = [m for m in runtime.session.messages if m["role"] == "tool"]
    assert "timed out" in tool_msgs[0]["content"]
    assert any(e == "tool.call" for e, _ in events)
    assert any(e == "tool.result" for e, _ in events)


def test_events_for_simple_turn():
    events: List[str] = []
    runtime = LLMRuntime(
        model="test/model",
        tools=[],
        tool_executor=DictToolExecutor({}),
        system_prompt_factory=lambda _ctx: "sys",
        completion_fn=ScriptedCompletion([_assistant_text("ok")]),
        on_event=lambda e, _p: events.append(e),
    )
    assert runtime.query("hi") == "ok"
    assert events == [
        "turn.start",
        "completion.start",
        "completion.end",
        "turn.end",
    ]


def test_validate_config_rejects_empty_model():
    with pytest.raises(ValueError, match="model"):
        validate_config({
            "model": "  ",
            "plugins": [],
            "max_tool_call_depth": 1,
            "api": {"kind": "completions"},
            "compactor": {"kind": "null"},
        })


def test_validate_config_rejects_bad_api_kind():
    with pytest.raises(ValueError, match="api.kind"):
        validate_config({
            "model": "test/model",
            "plugins": [],
            "max_tool_call_depth": 1,
            "api": {"kind": "graphql"},
            "compactor": {"kind": "null"},
        })


def test_validate_config_rejects_bad_timeout():
    with pytest.raises(ValueError, match="completion_timeout"):
        validate_config({
            "model": "test/model",
            "plugins": [],
            "max_tool_call_depth": 1,
            "api": {"kind": "completions"},
            "compactor": {"kind": "null"},
            "completion_timeout": 0,
        })


def test_validate_config_accepts_yaml_null_compactor_kind():
    # YAML ``kind: null`` becomes Python None.
    cfg = validate_config({
        "model": "test/model",
        "plugins": [],
        "max_tool_call_depth": 1,
        "api": {"kind": "completions"},
        "compactor": {"kind": None},
    })
    assert cfg["compactor"]["kind"] == "null"


def test_apply_tool_filters_allow_and_deny():
    schemas = [
        {"type": "function", "function": {"name": "a", "parameters": {}}},
        {"type": "function", "function": {"name": "b", "parameters": {}}},
        {"type": "function", "function": {"name": "c", "parameters": {}}},
    ]
    functions = {"a": lambda: 1, "b": lambda: 2, "c": lambda: 3}
    schemas2, functions2 = apply_tool_filters(schemas, functions, allow=["a", "b"], deny=["b"])
    assert sorted(functions2) == ["a"]
    assert [s["function"]["name"] for s in schemas2] == ["a"]


def test_apply_tool_filters_unknown_allow_raises():
    with pytest.raises(ValueError, match="tools.allow"):
        apply_tool_filters([], {"a": lambda: 1}, allow=["missing"], deny=[])


def test_build_runtime_applies_tool_deny():
    runtime = build_runtime({
        "model": "test/model",
        "plugins": ["time", "todo"],
        "max_tool_call_depth": 2,
        "compactor": {"kind": "null"},
        "api": {"kind": "completions"},
        "tools": {"allow": None, "deny": ["todo_clear", "todo_complete"]},
    })
    names = set(runtime.tool_executor.tools)
    assert "get_current_time" in names
    assert "todo_add" in names
    assert "todo_clear" not in names
    assert "todo_complete" not in names


def test_build_runtime_tool_allow_subset():
    runtime = build_runtime({
        "model": "test/model",
        "plugins": ["time", "todo"],
        "max_tool_call_depth": 2,
        "compactor": {"kind": "null"},
        "api": {"kind": "completions"},
        "tools": {"allow": ["get_current_time"], "deny": []},
    })
    assert set(runtime.tool_executor.tools) == {"get_current_time"}
