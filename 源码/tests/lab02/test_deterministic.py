import unittest
from core.kb.domain import demo_case, load_from_equity_kg, scaled_case
from core.kb.facts import build_facts
from core.kb.rules import build_rules, DERIVED_PREDS
from core.kb.rules_dsl import Pattern as P, Rule, RuleSet, all_of, gt, neq, eval_guard
from experiments.lab02.deterministic.forward import ForwardEngine, InferenceLimitError, match_pattern
from experiments.lab02.deterministic.backward import BackwardEngine


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = build_rules()
        cls.case = load_from_equity_kg()
        cls.facts = build_facts(cls.case).to_tuples()
        cls.forward = ForwardEngine(cls.rules, cls.facts)
        cls.forward.run()
        cls.backward = BackwardEngine(cls.rules, cls.facts)

    def test_original_regression(self):
        yes = [("Controls", "P1", "C3"), ("Controls", "P1", "C5"),
               ("Controls", "P3", "C6"), ("ActualController", "C3", "P1"),
               ("RelatedTrade", "C4", "C3"), ("Alert", "C3", "关联交易待核查"),
               ("RiskLevel", "C3", "高"), ("BeneficialOwner", "C3", "P1"),
               ("Alert", "C6", "循环持股"), ("Alert", "C7", "担保圈")]
        no = [("Controls", "P3", "C3"), ("Controls", "P3", "C12"),
              ("Controls", "P4", "C12"), ("ActualController", "C3", "P3"),
              ("RelatedTrade", "C3", "C4"), ("Alert", "C1", "关联交易待核查")]
        for truth, goals in [(True, yes), (False, no)]:
            for g in goals:
                with self.subTest(goal=g):
                    self.assertEqual(self.forward.holds(g), truth)
                    self.assertEqual(self.backward.solve(g) is not None, truth)

    def test_all_predicates_equivalent(self):
        for pred in sorted(DERIVED_PREDS):
            args = tuple(f"?v{i}" for i in range(self.rules._arity[pred]))
            with self.subTest(pred=pred):
                self.assertEqual(set(self.backward.ask(pred, *args)),
                                 set(self.forward.query(pred, *args)))

    def test_naive_matches_seminaive(self):
        baseline = ForwardEngine(self.rules, self.facts)
        baseline.run(semi_naive=False)
        self.assertEqual(baseline.wm.facts, self.forward.wm.facts)

    def test_extended_scenario(self):
        case = demo_case()
        self.assertEqual((len(case.persons), len(case.companies)), (9, 20))
        self.assertTrue(set(self.case.holdings) <= set(case.holdings))
        self.assertTrue(set(self.case.trades) <= set(case.trades))
        fw = ForwardEngine(self.rules, build_facts(case).to_tuples())
        fw.run()
        for goal in [("Controls", "P5", "C15"), ("Alert", "C16", "循环持股"),
                     ("Alert", "C19", "担保圈"), ("OpaqueShell", "P8", "C20")]:
            self.assertTrue(fw.holds(goal), goal)
        self.assertFalse(fw.holds(("Controls", "P5", "C3")))
        self.assertFalse(fw.holds(("Concerts", "P3", "P8")))

    def test_extended_all_predicates_equivalent(self):
        facts = build_facts(demo_case()).to_tuples()
        fw = ForwardEngine(self.rules, facts)
        fw.run()
        bw = BackwardEngine(self.rules, facts)
        for pred in sorted(DERIVED_PREDS):
            args = tuple(f"?v{i}" for i in range(self.rules._arity[pred]))
            with self.subTest(pred=pred):
                self.assertEqual(set(bw.ask(pred, *args)), set(fw.query(pred, *args)))

    def test_self_holding_is_cycle_not_self_control(self):
        facts = [("Holds", "c", "c", .6)]
        fw = ForwardEngine(self.rules, facts)
        fw.run()
        self.assertTrue(fw.holds(("Alert", "c", "循环持股")))
        self.assertFalse(fw.holds(("Controls", "c", "c")))
        self.assertIsNone(BackwardEngine(self.rules, facts).solve(("Controls", "c", "c")))

    def test_repeated_variable_and_existing_binding(self):
        self.assertIsNone(match_pattern(P.of("R", "?x", "?x"), ("R", "a", "b"), {}))
        self.assertIsNone(match_pattern(P.of("R", "?x"), ("R", "b"), {"?x": "a"}))

    def test_backtracks_all_premise_answers(self):
        rs = RuleSet([Rule("r", "", (P.of("Edge", "?x", "?y"), P.of("Good", "?y")), P.of("Yes", "?x"))])
        facts = [("Edge", "a", "b"), ("Edge", "a", "c"), ("Good", "c")]
        bw = BackwardEngine(rs, facts)
        proof = bw.solve(("Yes", "a"))
        self.assertIsNotNone(proof)
        self.assertEqual(proof.children[0].goal, ("Edge", "a", "c"))

    def test_left_recursion_cycles_and_fresh_variables(self):
        rs = RuleSet([
            Rule("t", "", (P.of("R", "?x", "?y"), P.of("E", "?y", "?z")), P.of("R", "?x", "?z")),
            Rule("b", "", (P.of("E", "?x", "?y"),), P.of("R", "?x", "?y"))])
        facts = [("E", "a", "b"), ("E", "b", "c"), ("E", "c", "a"), ("E", "c", "d")]
        fw = ForwardEngine(rs, facts)
        fw.run()
        bw = BackwardEngine(rs, facts)
        self.assertEqual(set(bw.ask("R", "?x", "?y")), set(fw.query("R", "?x", "?y")))
        self.assertEqual(set(bw.ask("R", "?x", "?x")), {( "R", x, x) for x in "abc"})
        self.assertIsNone(bw.solve(("R", "d", "a")))

    def test_empty_body_and_no_facts(self):
        rs = RuleSet([Rule("axiom", "", (), P.of("A"))])
        fw = ForwardEngine(rs, [])
        fw.run()
        self.assertTrue(fw.holds(("A",)))
        self.assertIsNotNone(BackwardEngine(rs, []).solve(("A",)))

    def test_guards_are_not_ignored(self):
        g = all_of(gt("?r", .5), neq("?x", "?y"))
        self.assertIsNone(eval_guard(g, {"?x": "a", "?y": "b"}))
        self.assertTrue(eval_guard(g, {"?x": "a", "?y": "b", "?r": .6}))
        self.assertFalse(eval_guard(g, {"?x": "a", "?y": "a", "?r": .6}))
        with self.assertRaises(ValueError):
            self.rules.for_head("DirectControls")[0].to_clauses()
        with self.assertRaises(ValueError):
            Rule("unsafe", "", (P.of("A", "?x"),), P.of("B", "?y"))

    def test_resource_limit_is_not_false(self):
        with self.assertRaises(InferenceLimitError):
            ForwardEngine(self.rules, self.facts).run(max_cycles=1)
        with self.assertRaises(InferenceLimitError):
            BackwardEngine(self.rules, self.facts, max_rounds=1).solve(("Controls", "P1", "C3"))

    def test_50_percent_is_not_logical_negation(self):
        f = [("Holds", "p", "c", .5), ("Holds", "q", "c", .1), ("Kin", "p", "q", "配偶")]
        fw = ForwardEngine(self.rules, f)
        fw.run()
        self.assertTrue(fw.holds(("NotControlsByHalf", "p", "c")))
        self.assertTrue(fw.holds(("Controls", "p", "c")))
        fw = ForwardEngine(self.rules, f[:1])
        fw.run()
        self.assertFalse(fw.holds(("Controls", "p", "c")))

    def test_scaled_valid_and_repeatable(self):
        self.assertEqual(scaled_case(6, 7), scaled_case(6, 7))
        scaled_case(20, 1).validate()

    def test_unknown_goal_explanation(self):
        self.assertIn("无法证明", self.forward.explain(("Controls", "P3", "C3"))[0])


if __name__ == "__main__":
    unittest.main()
