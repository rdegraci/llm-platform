"""Tests for the experimental Responses API adapter."""
from functools import partial
from typing import Any, Dict, List

import litellm
import pytest

from llm_platform.loader import completion_fn_for_config
from llm_platform.responses_adapter import (
    messages_to_responses_input,
    responses_completion,
    responses_output_to_completion,
    tools_to_responses_tools,
)
from llm_platform.runtime import DictToolExecutor, LLMRuntime


def test_tools_to_responses_tools_flattens_function_schema():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                },
            },
        }
    ]
    assert tools_to_responses_tools(tools) == [
        {
            "type": "function",
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            },
        }
    ]


def test_tools_to_responses_tools_skips_non_function_types():
    tools = [
        {"type": "web_search_preview"},
        {
            "type": "function",
            "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}},
        },
    ]
    converted = tools_to_responses_tools(tools)
    assert converted == [
        {"type": "function", "name": "ping", "parameters": {"type": "object", "properties": {}}}
    ]


def test_messages_to_responses_input_maps_system_tools_and_results():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ping", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "pong"},
        {"role": "assistant", "content": "done"},
    ]
    instructions, input_items = messages_to_responses_input(messages)
    assert instructions == "sys"
    assert input_items[0] == {"role": "user", "content": "hi"}
    assert input_items[1] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "ping",
        "arguments": "{}",
    }
    assert input_items[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "pong",
    }
    assert input_items[3] == {"role": "assistant", "content": "done"}


def test_responses_output_to_completion_text_and_function_calls():
    response = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello "}],
            },
            {
                "type": "function_call",
                "call_id": "call_9",
                "name": "add",
                "arguments": '{"a": 1}',
            },
        ]
    }
    mapped = responses_output_to_completion(response)
    message = mapped["choices"][0]["message"]
    assert message["content"] == "hello "
    assert message["tool_calls"] == [
        {
            "id": "call_9",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 1}'},
        }
    ]


class ScriptedResponses:
    """Fake litellm.responses: return scripted payloads, record kwargs."""

    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("ScriptedResponses exhausted")
        return self._responses.pop(0)


def _runtime_with_responses(script: ScriptedResponses, tools=None, max_depth: int = 3) -> LLMRuntime:
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
        max_tool_call_depth=max_depth,
        completion_fn=partial(responses_completion, responses_fn=script),
    )


def test_responses_adapter_no_tools_round_trip():
    script = ScriptedResponses([
        {"output": [{"type": "message", "content": [{"type": "output_text", "text": "4"}]}]},
    ])
    runtime = _runtime_with_responses(script)
    assert runtime.query("2+2") == "4"
    assert script.calls[0]["instructions"] == "sys"
    assert script.calls[0]["input"][0] == {"role": "user", "content": "2+2"}
    assert script.calls[0]["tools"] == []


def test_responses_adapter_successful_tool_round():
    script = ScriptedResponses([
        {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "add",
                    "arguments": '{"a": 2, "b": 3}',
                }
            ]
        },
        {"output": [{"type": "message", "content": [{"type": "output_text", "text": "5"}]}]},
    ])
    runtime = _runtime_with_responses(
        script,
        tools={"add": lambda a, b: a + b},
    )
    assert runtime.query("2+3") == "5"
    assert len(script.calls) == 2
    second_input = script.calls[1]["input"]
    assert {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "5",
    } in second_input


def test_responses_adapter_unknown_tool_continues():
    script = ScriptedResponses([
        {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_bad",
                    "name": "missing",
                    "arguments": "{}",
                }
            ]
        },
        {"output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]},
    ])
    runtime = _runtime_with_responses(script, tools={"ping": lambda: "pong"})
    assert runtime.query("go") == "ok"
    tool_msgs = [m for m in runtime.session.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "Unknown tool: 'missing'" in tool_msgs[0]["content"]


def test_responses_adapter_depth_limit_pairs_call_ids():
    script = ScriptedResponses([
        {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_a",
                    "name": "ping",
                    "arguments": "{}",
                }
            ]
        },
        {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_b",
                    "name": "ping",
                    "arguments": "{}",
                }
            ]
        },
    ])
    runtime = _runtime_with_responses(
        script,
        tools={"ping": lambda: "pong"},
        max_depth=1,
    )
    result = runtime.query("go")
    assert "Maximum tool call depth reached" in result
    tool_msgs = [m for m in runtime.session.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_a", "call_b"]


def test_completion_fn_for_config_defaults_to_completions():
    assert completion_fn_for_config({}) is litellm.completion
    assert completion_fn_for_config({"api": {"kind": "completions"}}) is litellm.completion


def test_completion_fn_for_config_responses_opt_in():
    assert completion_fn_for_config({"api": {"kind": "responses"}}) is responses_completion


def test_completion_fn_for_config_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown api.kind"):
        completion_fn_for_config({"api": {"kind": "graphql"}})
