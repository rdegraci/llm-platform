"""Generic LLM runtime components for tool-calling applications."""
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

import litellm

logger = logging.getLogger(__name__)

load_dotenv()

# Optional structured event hook: on_event(event_name, payload_dict)
EventCallback = Callable[[str, Dict[str, Any]], None]


def configure_litellm_api_key() -> Optional[str]:
    """Load environment variables and configure the litellm/OpenAI API key."""
    candidates = [
        "LITELLM_API_KEY",
        "OPENAI_API_KEY",
        "API_KEY",
        "LLM_API_KEY",
    ]

    found_key = None
    for name in candidates:
        val = os.getenv(name)
        if val:
            found_key = val
            break

    if found_key is not None:
        os.environ["LITELLM_API_KEY"] = found_key
        os.environ["OPENAI_API_KEY"] = found_key
        try:
            if hasattr(litellm, "api_key"):
                setattr(litellm, "api_key", found_key)
        except Exception:
            logger.exception("Failed to set litellm.api_key attribute")

    return found_key


def run_with_timeout(
    fn: Callable[..., Any],
    timeout: Optional[float],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run ``fn`` and raise ``TimeoutError`` if it exceeds ``timeout`` seconds.

    ``timeout`` of ``None`` or ``<= 0`` disables the limit. On timeout the worker
    thread is not killed; it may still finish in the background.
    """
    if timeout is None or timeout <= 0:
        return fn(*args, **kwargs)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as e:
            raise TimeoutError(
                f"Call exceeded timeout of {timeout} seconds"
            ) from e


def _preview(value: Any, limit: int = 500) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _get_field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def extract_message(response: Any) -> Any:
    """Extract a message object/dict from a completion response or message-like input."""
    if response is None:
        return None

    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, (list, tuple)) and choices:
            first = choices[0]
            if isinstance(first, dict) and "message" in first:
                return first.get("message")
            if isinstance(first, dict) and (
                "content" in first or "function_call" in first or "tool_calls" in first
            ):
                return first

    if hasattr(response, "choices"):
        try:
            choices = getattr(response, "choices")
            if choices:
                first = choices[0]
                if hasattr(first, "message"):
                    return getattr(first, "message")
                if hasattr(first, "content") or hasattr(first, "function_call") or hasattr(first, "tool_calls"):
                    return first
        except Exception:
            logger.exception("Exception while extracting message from attribute-style response")

    if isinstance(response, dict) and (
        "content" in response or "function_call" in response or "tool_calls" in response
    ):
        return response

    if hasattr(response, "content") or hasattr(response, "function_call") or hasattr(response, "tool_calls"):
        return response

    logger.debug("Response shape not recognized for extraction: %s", type(response))
    return None


def normalize_response(response: Any) -> Dict[str, Any]:
    """Normalize a completion response into a consistent message dictionary."""
    msg = extract_message(response)
    if msg is None:
        logger.warning("Unable to extract message from response of type %s", type(response))
        msg = {}

    content = _get_field(msg, "content")
    function_call = _get_field(msg, "function_call")
    tool_calls = _get_field(msg, "tool_calls")

    if tool_calls is None and function_call is not None:
        tool_calls = [function_call]
    if tool_calls is None:
        tool_calls = []

    return {
        "message": {
            "content": content,
            "function_call": function_call,
            "tool_calls": tool_calls,
            "raw": msg,
        },
        "raw": response,
    }


def message_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a message-like object to a plain dictionary."""
    if obj is None:
        return {}

    msg = {}
    for key in ("role", "content", "function_call", "tool_calls", "name", "id", "tool_call_id"):
        value = _get_field(obj, key)
        if value is not None:
            msg[key] = value
    return msg


class ConversationSession:
    """Mutable conversation state for a single LLM interaction session."""

    def __init__(self, system_prompt: str):
        self._system_prompt = system_prompt
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def reset(self, system_prompt: Optional[str] = None) -> None:
        if system_prompt is not None:
            self._system_prompt = system_prompt
        self.messages = [{"role": "system", "content": self._system_prompt}]

    def refresh_system_prompt(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = system_prompt
        else:
            self.messages.insert(0, {"role": "system", "content": system_prompt})

    def append(self, message: Dict[str, Any]) -> None:
        self.messages.append(message)


class DictToolExecutor:
    """Simple tool executor backed by a name-to-callable mapping."""

    def __init__(self, tools: Dict[str, Callable[..., Any]]):
        self._tools = dict(tools)

    @property
    def tools(self) -> Dict[str, Callable[..., Any]]:
        return self._tools

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def execute(self, name: str, args: Dict[str, Any]) -> Any:
        return self._tools[name](**args)


class Compactor:
    """Protocol for conversation-history compaction strategies."""

    def compact(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return messages


class NullCompactor(Compactor):
    """No-op compactor: returns messages unchanged."""

    def compact(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return messages


class SlidingWindowCompactor(Compactor):
    """Keep system messages plus the last N user/assistant turns.

    A turn starts at a user message and includes everything that follows
    until the next user message — so each tool-call cluster (assistant
    tool_calls + tool results + final assistant reply) stays attached to
    its originating user turn and never gets orphaned.
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns

    def compact(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if len(user_indices) <= self.max_turns:
            return messages
        cutoff = user_indices[-self.max_turns]
        preserved = [m for m in messages[:cutoff] if m.get("role") == "system"]
        return preserved + list(messages[cutoff:])


class LLMRuntime:
    """Generic tool-calling LLM runtime with injected prompt and tool layers."""

    def __init__(
        self,
        model: str,
        tools: List[Dict[str, Any]],
        tool_executor: DictToolExecutor,
        system_prompt_factory: Callable[[Any], str],
        user_context: Any = None,
        max_tool_call_depth: int = 3,
        completion_fn: Optional[Callable[..., Any]] = None,
        compactor: Optional[Compactor] = None,
        completion_timeout: Optional[float] = None,
        tool_timeout: Optional[float] = None,
        on_event: Optional[EventCallback] = None,
    ):
        self.model = model
        self.tools = tools
        self.tool_executor = tool_executor
        self.system_prompt_factory = system_prompt_factory
        self.user_context = user_context
        self.max_tool_call_depth = max_tool_call_depth
        self.completion_fn = completion_fn or litellm.completion
        self.compactor = compactor or NullCompactor()
        self.completion_timeout = completion_timeout
        self.tool_timeout = tool_timeout
        self.on_event = on_event
        self.session = ConversationSession(self.system_prompt_factory(self.user_context))

    def _emit(self, event: str, **payload: Any) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event, payload)
        except Exception:
            logger.exception("on_event handler failed for %s", event)

    def refresh_system_prompt(self, user_context: Any = None) -> None:
        if user_context is not None:
            self.user_context = user_context
        self.session.refresh_system_prompt(self.system_prompt_factory(self.user_context))

    def reset(self, user_context: Any = None) -> None:
        if user_context is not None:
            self.user_context = user_context
        self.session.reset(self.system_prompt_factory(self.user_context))

    def query(self, user_prompt: str) -> Any:
        self._emit("turn.start", user_prompt=_preview(user_prompt))
        try:
            self.session.append({"role": "user", "content": user_prompt})
            response = self._complete()
            result = self._continue_from_response(response, depth=0)
            self._emit("turn.end", result=_preview(result))
            return result
        except Exception as e:
            self._emit(
                "error",
                phase="turn",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def _complete(self) -> Any:
        self.session.messages = self.compactor.compact(self.session.messages)
        self._emit(
            "completion.start",
            model=self.model,
            message_count=len(self.session.messages),
        )
        try:
            response = run_with_timeout(
                self.completion_fn,
                self.completion_timeout,
                model=self.model,
                messages=self.session.messages,
                tools=self.tools,
            )
        except TimeoutError as e:
            self._emit(
                "error",
                phase="completion",
                error=str(e),
                error_type="TimeoutError",
            )
            raise
        self._emit("completion.end")
        return response

    def _continue_from_response(self, response: Any, depth: int) -> Any:
        norm = normalize_response(response)
        message = norm["message"]

        if message.get("tool_calls"):
            return self._handle_tool_calls(response, depth=depth)

        assistant_content = message.get("content")
        message_raw = message.get("raw")
        if message_raw is not None:
            assistant_msg = message_to_dict(message_raw)
            if "role" not in assistant_msg:
                assistant_msg["role"] = "assistant"
            if "content" not in assistant_msg:
                assistant_msg["content"] = assistant_content
            self.session.append(assistant_msg)
            return assistant_content

        self.session.append({"role": "assistant", "content": assistant_content})
        return assistant_content

    def _handle_tool_calls(self, response: Any, depth: int = 0) -> Any:
        norm = normalize_response(response)
        message = norm["message"]
        message_raw = message.get("raw")
        raw_calls = message.get("tool_calls") or []

        normalized: List[Dict[str, Any]] = []
        for tc in raw_calls:
            if not tc:
                continue
            nested = _get_field(tc, "function")
            if nested:
                if isinstance(nested, dict):
                    name = nested.get("name")
                    args_raw = nested.get("arguments", "{}")
                else:
                    name = _get_field(nested, "name")
                    args_raw = _get_field(nested, "arguments") or "{}"
            else:
                name = _get_field(tc, "name")
                args_raw = _get_field(tc, "arguments") or "{}"
            call_id = _get_field(tc, "id") or f"call_{uuid.uuid4().hex}"
            normalized.append({"id": call_id, "name": name, "arguments_raw": args_raw})

        if message_raw is not None:
            assistant_msg = message_to_dict(message_raw)
            # Rewrite legacy function_call to the modern tool_calls shape so the
            # conversation history stays uniform regardless of which shape the
            # model emitted.
            if "function_call" in assistant_msg and "tool_calls" not in assistant_msg:
                assistant_msg.pop("function_call")
                assistant_msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments_raw"]},
                    }
                    for c in normalized
                ]
            if "role" not in assistant_msg:
                assistant_msg["role"] = "assistant"
            self.session.append(assistant_msg)

        if depth >= self.max_tool_call_depth:
            warning_msg = (
                f"Maximum tool call depth reached ({self.max_tool_call_depth}). "
                "No further tool/function calls will be processed."
            )
            for call in normalized:
                self.session.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": warning_msg,
                })
            return warning_msg

        for call in normalized:
            name = call["name"]
            args_raw = call["arguments_raw"]
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                logger.exception("Failed to parse function arguments for %s", name)
                args = {}

            self._emit(
                "tool.call",
                tool=name,
                tool_call_id=call["id"],
                arguments=_preview(args),
            )

            if not self.tool_executor.has_tool(name):
                result = (
                    f"Unknown tool: {name!r}. "
                    f"Available: {sorted(self.tool_executor.tools)}"
                )
            else:
                try:
                    result = run_with_timeout(
                        self.tool_executor.execute,
                        self.tool_timeout,
                        name,
                        args,
                    )
                except TimeoutError as e:
                    self._emit(
                        "error",
                        phase="tool",
                        tool=name,
                        tool_call_id=call["id"],
                        error=str(e),
                        error_type="TimeoutError",
                    )
                    result = f"Function {name} timed out: {e}"
                except Exception as e:
                    logger.exception("Function %s raised an exception", name)
                    self._emit(
                        "error",
                        phase="tool",
                        tool=name,
                        tool_call_id=call["id"],
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    result = f"Function {name} raised an exception: {e}"

            self._emit(
                "tool.result",
                tool=name,
                tool_call_id=call["id"],
                result=_preview(result),
            )
            self.session.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            })

        second_response = self._complete()
        return self._continue_from_response(second_response, depth=depth + 1)
