"""Built-in plugin: session-scoped todo list (opt-in).

Items live on the plugin instance for the lifetime of one LLMRuntime.
They are not written to disk.
"""
from typing import Any, Callable, Dict, List

from llm_platform.plugin_api import Plugin, register_plugin


@register_plugin
class TodoPlugin(Plugin):
    """In-memory checklist for multi-step work within one session."""

    name = "todo"
    summary = "Track a session-scoped checklist of tasks."

    def __init__(self) -> None:
        self._items: List[Dict[str, Any]] = []
        self._next_id = 1

    def tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "todo_add",
                    "description": "Add a task to the session todo list. Returns the new item.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Short task description.",
                            }
                        },
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "todo_list",
                    "description": "List all session todo items (open and completed).",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "todo_complete",
                    "description": "Mark a todo item complete by id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "description": "Todo item id from todo_add or todo_list.",
                            }
                        },
                        "required": ["id"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "todo_clear",
                    "description": "Remove all todo items from this session.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def tool_functions(self) -> Dict[str, Callable[..., Any]]:
        return {
            "todo_add": self.todo_add,
            "todo_list": self.todo_list,
            "todo_complete": self.todo_complete,
            "todo_clear": self.todo_clear,
        }

    def system_prompt_section(self, user_context: Any = None) -> str:
        return (
            "## Todo\n\n"
            "Session-scoped checklist only. Items are not saved across restarts.\n"
            "Use todo_add, todo_list, and todo_complete for multi-step work. "
            "Do not build a long plan when a direct answer works."
        )

    def todo_add(self, text: str) -> Dict[str, Any]:
        item = {"id": self._next_id, "text": text, "done": False}
        self._next_id += 1
        self._items.append(item)
        return dict(item)

    def todo_list(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._items]

    def todo_complete(self, id: int) -> Dict[str, Any]:
        for item in self._items:
            if item["id"] == id:
                item["done"] = True
                return dict(item)
        return {"error": f"No todo item with id {id}"}

    def todo_clear(self) -> Dict[str, Any]:
        count = len(self._items)
        self._items.clear()
        return {"cleared": count}
