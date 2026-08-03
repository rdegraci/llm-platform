"""Tool-loop tests for LLMRuntime with a fake completion_fn."""
from typing import Any, Dict, List

from llm_platform.runtime import DictToolExecutor, LLMRuntime, SlidingWindowCompactor


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
    """Return scripted responses in order. Record each call for assertions."""

    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("ScriptedCompletion exhausted")
        return self._responses.pop(0)


def _runtime(
    completion_fn,
    tools=None,
    max_tool_call_depth: int = 3,
    compactor=None,
) -> LLMRuntime:
    tools = tools or {}
    schemas = [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in tools
    ]
    return LLMRuntime(
        model="test/model",
        tools=schemas,
        tool_executor=DictToolExecutor(tools),
        system_prompt_factory=lambda _ctx: "sys",
        max_tool_call_depth=max_tool_call_depth,
        completion_fn=completion_fn,
        compactor=compactor,
    )


def test_no_tools_returns_content_and_records_messages():
    script = ScriptedCompletion([_assistant_text("hello")])
    runtime = _runtime(script, tools={})

    assert runtime.query("hi") == "hello"
    roles = [m["role"] for m in runtime.session.messages]
    assert roles == ["system", "user", "assistant"]
    assert len(script.calls) == 1


def test_one_successful_tool_round():
    script = ScriptedCompletion([
        _assistant_tools([{"id": "call_1", "name": "add", "arguments": '{"a": 2, "b": 3}'}]),
        _assistant_text("5"),
    ])
    runtime = _runtime(script, tools={"add": lambda a, b: a + b})

    assert runtime.query("2+3") == "5"
    tool_msgs = [m for m in runtime.session.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_1"
    assert tool_msgs[0]["content"] == 5
    assert len(script.calls) == 2


def test_unknown_tool_appends_error_and_continues():
    script = ScriptedCompletion([
        _assistant_tools([
            {"id": "call_ok", "name": "ping", "arguments": "{}"},
            {"id": "call_bad", "name": "missing", "arguments": "{}"},
        ]),
        _assistant_text("done"),
    ])
    runtime = _runtime(script, tools={"ping": lambda: "pong"})

    assert runtime.query("go") == "done"
    tool_msgs = [m for m in runtime.session.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == "call_ok"
    assert tool_msgs[0]["content"] == "pong"
    assert tool_msgs[1]["tool_call_id"] == "call_bad"
    assert "Unknown tool: 'missing'" in tool_msgs[1]["content"]
    assert "ping" in tool_msgs[1]["content"]


def test_tool_exception_becomes_tool_content_and_continues():
    def boom():
        raise RuntimeError("nope")

    script = ScriptedCompletion([
        _assistant_tools([{"id": "call_1", "name": "boom", "arguments": "{}"}]),
        _assistant_text("recovered"),
    ])
    runtime = _runtime(script, tools={"boom": boom})

    assert runtime.query("go") == "recovered"
    tool_msgs = [m for m in runtime.session.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "raised an exception" in tool_msgs[0]["content"]
    assert "nope" in tool_msgs[0]["content"]


def test_depth_exceeded_pairs_tool_results_with_call_ids():
    # depth 0 executes; depth 1 hits max_tool_call_depth=1 and stops.
    script = ScriptedCompletion([
        _assistant_tools([{"id": "call_a", "name": "ping", "arguments": "{}"}]),
        _assistant_tools([
            {"id": "call_b", "name": "ping", "arguments": "{}"},
            {"id": "call_c", "name": "ping", "arguments": "{}"},
        ]),
    ])
    runtime = _runtime(script, tools={"ping": lambda: "pong"}, max_tool_call_depth=1)

    result = runtime.query("go")
    assert "Maximum tool call depth reached" in result

    tool_msgs = [m for m in runtime.session.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_a", "call_b", "call_c"]
    assert all("tool_call_id" in m for m in tool_msgs)
    assert "Maximum tool call depth reached" in tool_msgs[1]["content"]
    assert "Maximum tool call depth reached" in tool_msgs[2]["content"]
    # No further completion after depth stop.
    assert len(script.calls) == 2

    # Every assistant tool_calls id has a matching tool result.
    call_ids = set()
    for m in runtime.session.messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                call_ids.add(tc["id"])
    result_ids = {
        m["tool_call_id"]
        for m in runtime.session.messages
        if m.get("role") == "tool"
    }
    assert call_ids == result_ids


def test_bad_json_args_uses_empty_dict_and_continues():
    seen = {}

    def echo(**kwargs):
        seen["kwargs"] = kwargs
        return "ok"

    script = ScriptedCompletion([
        _assistant_tools([{"id": "call_1", "name": "echo", "arguments": "not-json"}]),
        _assistant_text("done"),
    ])
    runtime = _runtime(script, tools={"echo": echo})

    assert runtime.query("go") == "done"
    assert seen["kwargs"] == {}


def test_compactor_preserves_system_and_tool_cluster():
    script = ScriptedCompletion([
        _assistant_text("a0"),
        _assistant_tools([{"id": "call_1", "name": "ping", "arguments": "{}"}]),
        _assistant_text("final"),
    ])
    runtime = _runtime(
        script,
        tools={"ping": lambda: "pong"},
        compactor=SlidingWindowCompactor(max_turns=1),
    )

    assert runtime.query("u0") == "a0"
    assert runtime.query("u1") == "final"

    # After the second query, only the latest user turn (plus system) remains.
    user_msgs = [m for m in runtime.session.messages if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "u1"
    assert runtime.session.messages[0]["role"] == "system"
    tool_msgs = [m for m in runtime.session.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_1"
