"""Plugin: safely evaluate basic arithmetic for the model."""
from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Dict, List, Union

from llm_platform import Plugin, register_plugin

Number = Union[int, float]

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> Number:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    # Python 3.10 may still see ast.Num in some edge paths; keep Constant primary.
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(
        "Unsupported expression. Use numbers and + - * / // % ** and parentheses only."
    )


def calculate(expression: str) -> Dict[str, Any]:
    """Evaluate a basic arithmetic expression. Returns result or error."""
    text = (expression or "").strip()
    if not text:
        return {"error": "expression must be a non-empty string"}
    try:
        tree = ast.parse(text, mode="eval")
        value = _eval_node(tree)
    except Exception as e:
        return {"error": str(e), "expression": text}
    return {"expression": text, "result": value}


@register_plugin
class CalculatePlugin(Plugin):
    """Expose a safe calculator tool to the model."""

    name = "calculate"
    summary = "Evaluate basic arithmetic expressions safely."

    def tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": (
                        "Evaluate a basic arithmetic expression and return the "
                        "numeric result. Supports + - * / // % ** and parentheses. "
                        "Use this instead of doing hard math yourself."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Arithmetic expression, e.g. '(2 + 3) * 4'.",
                            }
                        },
                        "required": ["expression"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def tool_functions(self) -> Dict[str, Callable[..., Any]]:
        return {"calculate": calculate}

    def system_prompt_section(self, user_context: Any = None) -> str:
        return (
            "## Calculate\n\n"
            "For arithmetic, call the `calculate` tool with the expression. "
            "Do not invent results when the tool is available."
        )
