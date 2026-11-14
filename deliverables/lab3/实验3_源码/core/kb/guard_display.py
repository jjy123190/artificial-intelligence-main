"""Read-only guard descriptions from evaluator metadata or original lambda syntax.

Never evaluate source strings or maintain a second rule table. Unsupported or
unavailable legacy source is explicitly marked incomplete, not unconditional.
"""
from __future__ import annotations

import ast
import inspect
import json

from .rules_dsl import Guard, GuardSpec

_COMPARISONS = {ast.Gt: ">", ast.GtE: "≥", ast.Lt: "<", ast.LtE: "≤",
                ast.Eq: "=", ast.NotEq: "≠"}
_ARITHMETIC = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


def _expression(node: ast.AST, binding_name: str) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)):
        return json.dumps(node.value, ensure_ascii=False)
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == binding_name and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str) and node.slice.value.startswith("?")
            and len(node.slice.value) > 1):
        return node.slice.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
        return f"({_expression(node.left, binding_name)} {_ARITHMETIC[type(node.op)]} {_expression(node.right, binding_name)})"
    if isinstance(node, ast.Compare) and all(type(op) in _COMPARISONS for op in node.ops):
        parts = [_expression(node.left, binding_name)]
        for operator, right in zip(node.ops, node.comparators):
            parts.extend((_COMPARISONS[type(operator)], _expression(right, binding_name)))
        return " ".join(parts)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        operator = " ∧ " if isinstance(node.op, ast.And) else " ∨ "
        return "(" + operator.join(_expression(value, binding_name) for value in node.values) + ")"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"(-{_expression(node.operand, binding_name)})"
    raise ValueError("Unsupported legacy guard syntax")


def _legacy_conditions(guard: Guard) -> tuple[str, ...] | None:
    try:
        source, _ = inspect.findsource(guard)
        tree = ast.parse("".join(source))
        candidates = [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)
                      and node.lineno == guard.__code__.co_firstlineno]
        if len(candidates) != 1:
            return None
        node = candidates[0]
        if (len(node.args.args) != 1 or node.args.posonlyargs or node.args.kwonlyargs
                or node.args.vararg or node.args.kwarg):
            return None
        body = node.body.values if isinstance(node.body, ast.BoolOp) and isinstance(node.body.op, ast.And) else [node.body]
        return tuple(_expression(part, node.args.args[0].arg) for part in body)
    except (OSError, TypeError, SyntaxError, AttributeError, ValueError):
        return None


def describe_guard(guard: Guard | None) -> dict:
    """Return conditions for display, without calling or modifying the guard."""
    if guard is None:
        conditions = ()
    elif isinstance(guard, GuardSpec):
        conditions = guard.display_conditions
    else:
        conditions = _legacy_conditions(guard)
    return {"guard_conditions": list(conditions or ()), "guard_complete": conditions is not None}
