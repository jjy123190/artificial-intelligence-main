"""Finite, self-contained checks of the two *given* migration examples.
No lab03 search, learned model, Lean installation, network, or training is used.
Run without arguments to check in memory; use --output to save JSON explicitly.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable


class Rejected(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise Rejected(code, detail)


TERMS = frozenset({"a", "b", "c"})  # Three symbols of the same carrier, not numbers.
CONTEXT = {"h1": ("a", "b"), "h2": ("b", "c")}
THEOREM_GOAL = ("c", "a")


def equality(value) -> tuple[str, str]:
    require(isinstance(value, (list, tuple)) and len(value) == 2,
            "bad_equality", repr(value))
    require(all(isinstance(v, str) and v in TERMS for v in value),
            "bad_term", repr(value))
    return tuple(value)


def check_certificate(node: dict, context: dict) -> tuple[str, str]:
    """Synthesize a conclusion from an explicit Eq fragment certificate.

    All premises are checked recursively. No assertion is trusted from the
    certificate itself; no unknown lemma, hole, or unsupported tactic is allowed.
    """
    require(isinstance(node, dict), "bad_certificate", "node must be an object")
    rule = node.get("rule")
    schema = {
        "hyp": {"rule", "ref"},
        "refl": {"rule", "term"},
        "symm": {"rule", "premise"},
        "trans": {"rule", "left", "right"},
    }
    require(rule in schema, "unknown_rule", repr(rule))
    require(set(node) == schema[rule], "certificate_shape",
            f"{rule} expects {sorted(schema[rule])}, got {sorted(node)}")
    if rule == "hyp":
        ref = node["ref"]
        require(isinstance(ref, str) and ref in context,
                "unknown_hypothesis", repr(ref))
        return equality(context[ref])
    if rule == "refl":
        return equality((node["term"], node["term"]))
    if rule == "symm":
        a, b = check_certificate(node["premise"], context)
        return b, a
    a, b = check_certificate(node["left"], context)
    c, d = check_certificate(node["right"], context)
    require(b == c, "trans_middle_mismatch", f"{a}={b} cannot compose with {c}={d}")
    return a, d


def verify_theorem(node: dict, expected=THEOREM_GOAL) -> dict:
    inferred = check_certificate(node, CONTEXT)
    require(inferred == equality(expected), "wrong_conclusion", repr(inferred))
    return {"accepted": True, "conclusion": list(inferred)}


PROOF_ACTIONS = [
    {"rule": "symm"},
    {"rule": "trans", "via": "b"},
    {"rule": "exact", "ref": "h1"},
    {"rule": "exact", "ref": "h2"},
]


def replay_proof(actions: list[dict]) -> dict:
    """Replay backward actions while constructing a certificate with all holes.

    The leftmost obligation is processed. A trans step creates two AND goals;
    only closing *both* allows the final certificate to be returned.
    """
    root = {"hole": True}
    pending = [(root, THEOREM_GOAL)]
    trace = [{"step": 0, "action": "initial", "open_goals": [list(THEOREM_GOAL)]}]
    for i, action in enumerate(actions, 1):
        require(bool(pending), "extra_action", str(i))
        slot, (a, b) = pending.pop(0)
        rule = action.get("rule")
        if rule == "symm":
            require(set(action) == {"rule"}, "action_shape", str(action))
            child = {"hole": True}
            slot.clear(); slot.update(rule="symm", premise=child)
            pending.insert(0, (child, (b, a)))
        elif rule == "trans":
            require(set(action) == {"rule", "via"}, "action_shape", str(action))
            via = action["via"]
            equality((a, via)); equality((via, b))
            left, right = {"hole": True}, {"hole": True}
            slot.clear(); slot.update(rule="trans", left=left, right=right)
            pending[0:0] = [(left, (a, via)), (right, (via, b))]
        elif rule == "exact":
            require(set(action) == {"rule", "ref"}, "action_shape", str(action))
            leaf = {"rule": "hyp", "ref": action["ref"]}
            require(check_certificate(leaf, CONTEXT) == (a, b),
                    "exact_type_mismatch", f"{action['ref']} does not prove {a}={b}")
            slot.clear(); slot.update(leaf)
        else:
            raise Rejected("unknown_action", repr(rule))
        trace.append({"step": i, "action": deepcopy(action),
                      "open_goals": [list(goal) for _, goal in pending]})
    require(not pending, "open_obligations", repr([goal for _, goal in pending]))
    result = verify_theorem(root)
    return {**result, "context": {k: list(v) for k, v in CONTEXT.items()},
            "trace": trace, "certificate": root,
            "backward_actions": len(actions), "nontrivial_equality_inferences": 2,
            "hypothesis_leaves": 2,
            "scope": "finite equality fragment; not a Lean kernel run"}


Fact = tuple[str, ...]
BLOCKS = frozenset({"A", "B", "C"})
ARITIES = {"ontable": 1, "clear": 1, "holding": 1, "on": 2, "handempty": 0}


def f(name: str, *args: str) -> Fact:
    return (name, *args)


def show_fact(fact: Fact) -> str:
    return fact[0] if len(fact) == 1 else f"{fact[0]}({','.join(fact[1:])})"


def show_state(state: frozenset[Fact]) -> list[str]:
    order = {"ontable": 0, "on": 1, "clear": 2, "holding": 3, "handempty": 4}
    return [show_fact(x) for x in sorted(state, key=lambda x: (order[x[0]], x[1:]))]


def check_state(state: frozenset[Fact]) -> None:
    """Validate the explicit single-gripper Blocks World invariant set.

    A held block is deliberately not clear in this encoding; Clear becomes true
    again on PutDown/Stack. States are complete closed-world fact sets.
    """
    for fact in state:
        require(fact[0] in ARITIES and len(fact) == ARITIES[fact[0]] + 1,
                "invalid_fact", repr(fact))
        require(all(arg in BLOCKS for arg in fact[1:]), "unknown_block", repr(fact))
    held = [x for x in BLOCKS if f("holding", x) in state]
    require(len(held) <= 1, "multiple_held", str(held))
    require((f("handempty") in state) == (len(held) == 0),
            "hand_inconsistent", str(held))
    support = {x: [y for y in BLOCKS if f("on", x, y) in state] for x in BLOCKS}
    for x in BLOCKS:
        count = int(f("ontable", x) in state) + int(x in held) + len(support[x])
        require(count == 1, "location_inconsistent", x)
        above = [y for y in BLOCKS if f("on", y, x) in state]
        require(len(above) <= 1, "multiple_above", x)
        require(not (x in held and above), "held_tower", x)
        require((f("clear", x) in state) == (not above and x not in held),
                "clear_inconsistent", x)
        seen, cur = set(), x
        while support[cur]:
            require(cur not in seen, "support_cycle", x)
            seen.add(cur)
            cur = support[cur][0]


@dataclass(frozen=True)
class Action:
    name: str
    args: tuple[str, ...]
    pre: frozenset[Fact]
    add: frozenset[Fact]
    delete: frozenset[Fact]
    cost: int = 1

    @property
    def label(self) -> str:
        return f"{self.name}({','.join(self.args)})"


def action(name: str, *args: str) -> Action:
    arity = {"PickUp": 1, "PutDown": 1, "Stack": 2, "Unstack": 2}
    require(name in arity and len(args) == arity[name], "action_shape", f"{name}{args}")
    require(all(x in BLOCKS for x in args), "unknown_block", repr(args))
    if len(args) == 2:
        require(args[0] != args[1], "same_block_arguments", repr(args))
    x = args[0]
    if name == "PickUp":
        pre = {f("ontable", x), f("clear", x), f("handempty")}
        add = {f("holding", x)}
        delete = pre
    elif name == "PutDown":
        pre = {f("holding", x)}
        add = {f("ontable", x), f("clear", x), f("handempty")}
        delete = pre
    elif name == "Stack":
        y = args[1]
        pre = {f("holding", x), f("clear", y)}
        add = {f("on", x, y), f("clear", x), f("handempty")}
        delete = pre
    else:
        y = args[1]
        pre = {f("on", x, y), f("clear", x), f("handempty")}
        add = {f("holding", x), f("clear", y)}
        delete = pre
    return Action(name, args, frozenset(pre), frozenset(add), frozenset(delete))


INITIAL = frozenset({f("ontable", x) for x in BLOCKS}
                    | {f("clear", x) for x in BLOCKS} | {f("handempty")})
GOAL = frozenset({f("on", "A", "B"), f("on", "B", "C"), f("handempty")})
PLAN = [("PickUp", "B"), ("Stack", "B", "C"), ("PickUp", "A"), ("Stack", "A", "B")]


def verify_plan(plan: list[tuple[str, ...]]) -> dict:
    state, cost = INITIAL, 0
    check_state(state)
    trace = [{"step": 0, "action": "initial", "state": show_state(state), "g": 0,
              "goal_satisfied": GOAL <= state}]
    for i, call in enumerate(plan, 1):
        op = action(*call)
        missing = op.pre - state
        require(not missing, "precondition_failed",
                f"step {i}: {op.label}; missing={show_state(frozenset(missing))}")
        state = frozenset((state - op.delete) | op.add)
        check_state(state)
        cost += op.cost
        trace.append({"step": i, "action": op.label, "preconditions": show_state(op.pre),
                      "add": show_state(op.add), "delete": show_state(op.delete),
                      "state": show_state(state), "g": cost, "goal_satisfied": GOAL <= state})
    require(GOAL <= state, "goal_not_satisfied", str(show_state(frozenset(GOAL - state))))
    # Specific analytic lower bound, not a generic planner or heuristic claim:
    # two new On facts -> two Stack actions; empty initial hand and consumption
    # of Holding by each Stack -> two separate acquiring actions, hence >=4.
    new_on = [fact for fact in GOAL - INITIAL if fact[0] == "on"]
    require(f("handempty") in INITIAL and len(new_on) == 2,
            "lower_bound_assumptions", "specific example changed")
    lower_bound = 2 * len(new_on)
    return {"accepted": True, "initial": show_state(INITIAL), "goal": show_state(GOAL),
            "trace": trace, "plan_length": len(plan), "plan_cost": cost,
            "analytic_lower_bound": lower_bound, "matches_lower_bound": cost == lower_bound,
            "lower_bound_reason": "two required new On facts each need a Stack and a separate acquisition; initially empty hand; unit costs",
            "scope": "given deterministic closed-world single-gripper plan; no search benchmark"}


def rejection_case(name: str, expected_code: str, call: Callable[[], object]) -> dict:
    try:
        call()
    except Rejected as exc:
        if exc.code != expected_code:
            raise AssertionError(f"{name}: expected {expected_code}, got {exc.code}") from exc
        return {"name": name, "rejected": True, "code": exc.code, "detail": exc.detail}
    raise AssertionError(f"{name}: invalid candidate was accepted")


def run_checks() -> dict:
    theorem = replay_proof(PROOF_ACTIONS)
    plan = verify_plan(PLAN)
    wrong_middle = {"rule": "trans", "left": {"rule": "hyp", "ref": "h2"},
                    "right": {"rule": "hyp", "ref": "h1"}}
    fabricated = deepcopy(theorem["certificate"])
    fabricated["premise"]["right"]["ref"] = "h3"
    missing = deepcopy(theorem["certificate"])
    del missing["premise"]["right"]
    negatives = [
        rejection_case("proof_reversed_middle", "trans_middle_mismatch",
                       lambda: verify_theorem(wrong_middle)),
        rejection_case("proof_fabricated_hypothesis", "unknown_hypothesis",
                       lambda: verify_theorem(fabricated)),
        rejection_case("proof_missing_AND_branch", "certificate_shape",
                       lambda: verify_theorem(missing)),
        rejection_case("proof_unfinished_obligation", "open_obligations",
                       lambda: replay_proof(PROOF_ACTIONS[:-1])),
        rejection_case("plan_stack_without_holding", "precondition_failed",
                       lambda: verify_plan([("Stack", "A", "B")])),
        rejection_case("plan_pick_with_occupied_hand", "precondition_failed",
                       lambda: verify_plan([("PickUp", "B"), ("PickUp", "C")])),
        rejection_case("plan_goal_incomplete", "goal_not_satisfied",
                       lambda: verify_plan(PLAN[:2])),
        rejection_case("plan_stack_on_itself", "same_block_arguments",
                       lambda: verify_plan([("PickUp", "A"), ("Stack", "A", "A")])),
    ]
    return {
        "schema_version": 1,
        "status": "passed",
        "scope": "two constructed examples and eight adversarial validation cases; not a migrated solver evaluation",
        "external_proof_kernel_executed": False,
        "lab03_search_executed": False,
        "llm_calls": 0,
        "training_runs": 0,
        "positive_examples_accepted": 2,
        "negative_cases_rejected": len(negatives),
        "theorem": theorem,
        "planning": plan,
        "negative_cases": negatives,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="explicit optional JSON output; otherwise read-only")
    args = parser.parse_args()
    result = run_checks()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"theorem", "planning", "negative_cases"}},
                     ensure_ascii=False, indent=2))
    print("Theorem: c=a; four backward actions, two equality rules and two hypothesis leaves.")
    print(f"Planning: {result['planning']['plan_length']} actions; cost={result['planning']['plan_cost']}; analytic lower bound=4.")
    for item in result["negative_cases"]:
        print(f"REJECTED {item['name']}: {item['code']}")


if __name__ == "__main__":
    main()
