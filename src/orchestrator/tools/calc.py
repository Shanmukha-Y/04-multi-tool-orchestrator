"""Calculator tool. Evaluates arithmetic expressions through a small
whitelisted AST walker rather than ``eval`` - a planner-composed string is
untrusted input, even coming from our own local model."""

from __future__ import annotations

import ast
import asyncio
import operator

from orchestrator.manifests import ScopeClass
from orchestrator.registry import tool_def

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class UnsafeExpressionError(ValueError):
    pass


def safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](_eval(node.operand))
        raise UnsafeExpressionError(f"disallowed expression element: {ast.dump(node)}")

    return _eval(tree)


@tool_def(
    name="calc",
    description="Arithmetic calculator (safe AST-based evaluator, no external calls).",
    capabilities=["math.calculate"],
    scope=ScopeClass.READ,
    priority=1,
    timeout_s=3.0,
    param_schema={"expression": "arithmetic expression, e.g. '(3 + 4) * 2'"},
)
async def calc(expression: str) -> dict:
    await asyncio.sleep(0)  # keep it a coroutine; this tool is CPU-only
    try:
        result = safe_eval(expression)
    except (UnsafeExpressionError, SyntaxError, ZeroDivisionError, TypeError) as exc:
        return {"expression": expression, "error": str(exc)}
    return {"expression": expression, "result": result}
