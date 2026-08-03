"""Built-in plugin: current time (opt-in)."""
from datetime import datetime
from typing import Any, Callable, Dict, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from llm_platform.plugin_api import Plugin, register_plugin


def get_current_time(timezone: str = "UTC") -> str:
    """Return the current time as an ISO-8601 timestamp in the given timezone."""
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return (
            f"Unknown timezone: {timezone!r}. "
            "Use an IANA name such as 'UTC' or 'America/New_York'."
        )
    except Exception as e:
        return f"Could not resolve timezone {timezone!r}: {e}"
    return datetime.now(zone).isoformat()


@register_plugin
class TimePlugin(Plugin):
    """Expose the current clock time to the model."""

    name = "time"
    summary = "Get the current date and time in a given timezone."

    def tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": (
                        "Return the current date and time as an ISO-8601 "
                        "timestamp. Use for precise 'now'; the system prompt "
                        "already states today's calendar date."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": (
                                    "IANA timezone name (default: UTC). "
                                    "Examples: UTC, America/New_York, Europe/London."
                                ),
                                "default": "UTC",
                            }
                        },
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def tool_functions(self) -> Dict[str, Callable[..., Any]]:
        return {"get_current_time": get_current_time}
