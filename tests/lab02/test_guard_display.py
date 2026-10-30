"""Guard descriptions must be complete, read-only and sourced from executable rules."""
from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from core.kb.guard_display import describe_guard
from core.kb.rules import build_rules
from core.kb.rules_dsl import (GuardSpec, all_of, approximately, eq_var, equals,
                               eval_guard, ge, ge_var, gt, gt_var, neq,
                               ratio_sum_gt, ratio_sum_le)
from experiments.lab01.reasoning.rules import RULES


class GuardDisplayTests(unittest.TestCase):
    def test_scalar_helpers_share_parameters_with_display(self):
        cases = [
            (gt("?r", 0.5), "?r > 0.5", {"?r": 0.8}, True),
            (ge("?r", 0.5), "?r ≥ 0.5", {"?r": 0.5}, True),
            (ge_var("?a", "?b"), "?a ≥ ?b", {"?a": 1, "?b": 1}, True),
            (gt_var("?a", "?b"), "?a > ?b", {"?a": 1, "?b": 1}, False),
            (neq("?a", "?b"), "?a ≠ ?b", {"?a": "P1", "?b": "P1"}, False),
            (eq_var("?a", "?b"), "?a = ?b", {"?a": "P1", "?b": "P1"}, True),
            (equals("?rel", "配偶"), '?rel = "配偶"', {"?rel": "配偶"}, True),
            (ratio_sum_gt(("?r1", "?r2"), 0.5), "?r1 + ?r2 > 0.5", {"?r1": 0.3, "?r2": 0.25}, True),
            (ratio_sum_le(("?r1", "?r2"), 0.5), "?r1 + ?r2 ≤ 0.5", {"?r1": 0.3, "?r2": 0.25}, False),
            (approximately("?r", 0.5), "abs(?r - 0.5) ≤ 1e-09", {"?r": 0.5}, True),
        ]
        for guard, text, binding, wanted in cases:
            with self.subTest(text=text):
                self.assertEqual(describe_guard(guard), {"guard_conditions": [text], "guard_complete": True})
                self.assertIs(eval_guard(guard, binding), wanted)

    def test_all_formal_rule_conditions_are_exposed(self):
        expected = {
            "R01": ["?r > 0.5", "?x ≠ ?y"],
            "R02": ['?rel = "配偶"'],
            "R04": ["?r1 ≥ ?r2", "?r1 + ?r2 > 0.5", "?p ≠ ?q"],
            "R05": ["?r1 + ?r2 > 0.5", "?x ≠ ?a", "?x ≠ ?c"],
            "R05b": ["?r1 > 0.5", "?x ≠ ?a", "?x ≠ ?c"],
            "R06": ["abs(?r - 0.5) ≤ 1e-09"],
            "R07": ["?x ≠ ?z"],
            "R14": ["?a ≠ ?b"],
        }
        for rule in build_rules():
            self.assertEqual(describe_guard(rule.guard), {"guard_conditions": expected.get(rule.name, []), "guard_complete": True}, rule.name)
        self.assertEqual(len(expected), sum(rule.guard is not None for rule in build_rules()))

    def test_original_lambdas_are_described_without_replacing_them(self):
        expected = {
            "R1": ["?r > 0.5"],
            "R2": ['?rel = "配偶"'],
            "R3": ["?r1 ≥ ?r2", "(?r1 + ?r2) > 0.5"],
            "R4": ["(?r1 + ?r2) > 0.5"],
            "R5": ["?x ≠ ?z"],
        }
        before = [(rule, rule.test) for rule in RULES]
        for rule in RULES:
            self.assertEqual(describe_guard(rule.test), {"guard_conditions": expected.get(rule.name, []), "guard_complete": True}, rule.name)
        self.assertTrue(all(rule.test is original for rule, original in before))
        self.assertFalse(RULES[0].test({"?r": 0.5}))
        self.assertTrue(RULES[0].test({"?r": 0.8}))

    def test_describing_guard_never_calls_predicate(self):
        predicate = Mock(side_effect=AssertionError("Must not evaluate during rendering"))
        guard = GuardSpec(frozenset(["?x"]), predicate, ("?x > 0",))
        self.assertTrue(describe_guard(guard)["guard_complete"])
        predicate.assert_not_called()

    def test_unknown_conditions_are_not_misrepresented_as_complete(self):
        guard = GuardSpec(frozenset(["?x"]), lambda b: b["?x"] > 0)
        self.assertEqual(describe_guard(guard), {"guard_conditions": [], "guard_complete": False})
        combined = all_of(gt("?x", 0), guard)
        self.assertEqual(describe_guard(combined), {"guard_conditions": [], "guard_complete": False})
        self.assertEqual(describe_guard(None), {"guard_conditions": [], "guard_complete": True})

    def test_unavailable_original_source_is_explicitly_incomplete(self):
        with patch("core.kb.guard_display.inspect.findsource", side_effect=OSError("Source unavailable")):
            self.assertFalse(describe_guard(RULES[0].test)["guard_complete"])

    def test_original_arithmetic_precedence_is_retained(self):
        guard = lambda b: (b["?a"] + b["?b"]) * 2 > 0.5
        self.assertEqual(describe_guard(guard)["guard_conditions"], ["((?a + ?b) * 2) > 0.5"])

    def test_unsupported_callback_is_not_guessed(self):
        guard = lambda b: b.get("?r", 0) > 0.5
        self.assertFalse(describe_guard(guard)["guard_complete"])


if __name__ == "__main__":
    unittest.main()
