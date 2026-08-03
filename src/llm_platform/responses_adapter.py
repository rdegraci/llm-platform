"""Experimental OpenAI Responses API adapter.

Maps Completions-shaped ``messages`` / ``tools`` to ``litellm.responses`` and
maps the Responses output back to a Completions-like response so
:class:`~llm_platform.runtime.LLMRuntime` can keep a Completions-shaped session.

v1 supports function tools only. Built-in Responses tools (web search, code
interpreter, reasoning items, and similar) are out of scope.

This path is opt-in via ``api.kind: responses`` in config. Completions remains
the default and supported transport.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import litellm

logger = logging.getLogger(__name__)


def _get_field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _flatten_content(content: Any) -> str:
    """Flatten Responses message content (string or list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            text = _get_field(part, "text")
            if text:
                parts.append(text)
        return "".join(parts)
    return str(content)


def _stringify_tool_output(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    except TypeError:
        return str(content)


def tools_to_responses_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Convert Chat Completions tool schemas to Responses function tools.

    Completions shape::

        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    Responses shape::

        {"type": "function", "name": ..., "description": ..., "parameters": ...}

    Non-function tool types are skipped (v1 is function tools only).
    """
    if tools is None:
        return None
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        if not tool:
            continue
        if tool.get("type") not in (None, "function"):
            logger.warning(
                "Skipping non-function tool type %r in Responses adapter (v1 function tools only).",
                tool.get("type"),
            )
            continue
        nested = tool.get("function")
        if isinstance(nested, dict):
            item: Dict[str, Any] = {"type": "function", "name": nested.get("name")}
            if "description" in nested:
                item["description"] = nested["description"]
            if "parameters" in nested:
                item["parameters"] = nested["parameters"]
            converted.append(item)
        elif "name" in tool:
            # Already Responses-shaped (or flat).
            item = {"type": "function", "name": tool["name"]}
            if "description" in tool:
                item["description"] = tool["description"]
            if "parameters" in tool:
                item["parameters"] = tool["parameters"]
            converted.append(item)
        else:
            logger.warning("Skipping tool without a function name: %r", tool)
    return converted


def messages_to_responses_input(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Map Completions messages to Responses ``instructions`` + ``input`` items."""
    instructions_parts: List[str] = []
    input_items: List[Dict[str, Any]] = []
    index = 0
    while index < len(messages) and messages[index].get("role") == "system":
        instructions_parts.append(messages[index].get("content") or "")
        index += 1
    instructions = "\n\n".join(instructions_parts) if instructions_parts else None

    for msg in messages[index:]:
        role = msg.get("role")
        if role == "system":
            input_items.append({"role": "system", "content": msg.get("content") or ""})
        elif role == "user":
            input_items.append({"role": "user", "content": msg.get("content") or ""})
        elif role == "assistant":
            content = msg.get("content")
            tool_calls = msg.get("tool_calls") or []
            if content:
                input_items.append({"role": "assistant", "content": content})
            for tc in tool_calls:
                if not tc:
                    continue
                nested = _get_field(tc, "function")
                if isinstance(nested, dict):
                    name = nested.get("name")
                    arguments = nested.get("arguments", "{}")
                else:
                    name = _get_field(tc, "name")
                    arguments = _get_field(tc, "arguments") or "{}"
                call_id = _get_field(tc, "id") or _get_field(tc, "call_id")
                input_items.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                })
            if not content and not tool_calls:
                input_items.append({"role": "assistant", "content": ""})
        elif role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id"),
                "output": _stringify_tool_output(msg.get("content")),
            })
        else:
            logger.warning("Skipping unsupported message role %r in Responses adapter.", role)

    return instructions, input_items


def responses_output_to_completion(response: Any) -> Dict[str, Any]:
    """Map a Responses API result to a Completions-like response dict."""
    output = _get_field(response, "output") or []
    content_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for item in output:
        item_type = _get_field(item, "type")
        if item_type == "function_call":
            call_id = _get_field(item, "call_id") or _get_field(item, "id")
            arguments = _get_field(item, "arguments") or "{}"
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": _get_field(item, "name"),
                    "arguments": arguments,
                },
            })
        elif item_type in ("message", "output_message"):
            content_parts.append(_flatten_content(_get_field(item, "content")))
        elif item_type in ("output_text", "text"):
            text = _get_field(item, "text")
            if text:
                content_parts.append(text)
        elif item_type is None and _get_field(item, "role") == "assistant":
            content_parts.append(_flatten_content(_get_field(item, "content")))
        # Skip reasoning and other non-function item types in v1.

    content: Optional[str]
    if content_parts:
        content = "".join(content_parts)
    else:
        output_text = _get_field(response, "output_text")
        content = output_text if output_text else None

    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        if not content:
            message["content"] = None

    return {
        "choices": [{"message": message}],
        "raw_responses": response,
    }


def responses_completion(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    responses_fn: Optional[Callable[..., Any]] = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    """Completions-compatible callable backed by ``litellm.responses``.

    Signature matches what :class:`~llm_platform.runtime.LLMRuntime` passes to
    ``completion_fn`` (``model``, ``messages``, ``tools``). Extra Completions
    kwargs are ignored.

    Pass ``responses_fn`` in tests to inject a fake client. Production uses
    ``litellm.responses``.
    """
    fn = responses_fn or litellm.responses
    instructions, input_items = messages_to_responses_input(messages)
    resp_tools = tools_to_responses_tools(tools)

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": input_items,
    }
    if instructions is not None:
        kwargs["instructions"] = instructions
    if resp_tools is not None:
        kwargs["tools"] = resp_tools

    response = fn(**kwargs)
    return responses_output_to_completion(response)
