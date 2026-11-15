import itertools
import unittest
from core.kb.features import FEATURES, RiskObservation, from_case
from core.kb.domain import load_from_equity_kg
from core.kb.facts import build_facts
from core.kb.rules import build_rules
from core.kb.screening import ScreeningRules
from experiments.lab02.deterministic.forward import ForwardEngine
from experiments.lab02.uncertain.bayes import BinaryNode, BinaryBayesNet, ImpossibleEvidence, RiskBayes
from experiments.lab02.uncertain.fuzzy import MamdaniFuzzy, triangular
from experiments.lab02.uncertain.dempster_shafer import Mass, TotalConflict, dempster, yager, fuse
from experiments.lab02.hybrid.model import HybridModel


def obs(value, id_="c"):
    return RiskObservation(id_, {f: value for f in FEATURES})


class UncertainTests(unittest.TestCase):
    def test_hand_computed_bayes(self):
        net = BinaryBayesNet([BinaryNode("B", ("A",), {(0,): .1, (1,): .8}), BinaryNode("A", (), {(): .2})])
        for method in ["ve", "enumeration"]:
            self.assertAlmostEqual(net.query("A", {"B": 1}, method=method), 2 / 3)
            self.assertAlmostEqual(net.query("B", {}, method=method), .24)
            self.assertEqual(net.query("A", {"A": 1}, method=method), 1)

    def test_generic_ve_matches_enumeration(self):
        net = BinaryBayesNet([
            BinaryNode("A", (), {(): .3}), BinaryNode("B", (), {(): .6}),
            BinaryNode("C", ("A", "B"), {(0, 0): .1, (0, 1): .4, (1, 0): .7, (1, 1): .95})])
        for states in itertools.product([None, 0, 1], repeat=3):
            ev = {n: v for n, v in zip("ABC", states) if v is not None}
            for query in "ABC":
                self.assertAlmostEqual(net.query(query, ev), net.query(query, ev, method="enumeration"))

    def test_star_analytic_matches_ve(self):
        model = RiskBayes()
        for states in itertools.product([None, .1, .9], repeat=4):
            sample = RiskObservation("test", dict(zip(FEATURES, states)))
            self.assertAlmostEqual(model.predict(sample), model.predict(sample, method="ve"), places=12)
        self.assertAlmostEqual(model.predict(obs(None)), model.prior)

    def test_network_validation_and_impossible_evidence(self):
        with self.assertRaises(ValueError):
            BinaryNode("A", ("B",), {(0,): .2})
        with self.assertRaises(ValueError):
            BinaryBayesNet([BinaryNode("A", ("B",), {(0,): .2, (1,): .8}),
                            BinaryNode("B", ("A",), {(0,): .1, (1,): .7})])
        net = BinaryBayesNet([BinaryNode("A", (), {(): 0})])
        for method in ["ve", "enumeration"]:
            with self.assertRaises(ImpossibleEvidence):
                net.query("A", {"A": 1}, method=method)

    def test_fit_missing_counts_and_serialization(self):
        rows = [obs(.9, "a"), obs(.1, "b"), obs(None, "c")]
        model = RiskBayes.fit(rows, [1, 0, 1])
        self.assertAlmostEqual(model.prior, 3 / 5)
        self.assertAlmostEqual(model.cpt["opacity"][1], 2 / 3)
        self.assertAlmostEqual(model.cpt["opacity"][0], 1 / 3)
        restored = RiskBayes.from_dict(model.as_dict())
        self.assertEqual(model.predict(rows[0]), restored.predict(rows[0]))

    def test_fuzzy_boundaries_and_missing(self):
        self.assertEqual(triangular(0, 0, 0, .5), 1)
        self.assertEqual(triangular(1, .5, 1, 1), 1)
        self.assertEqual(triangular(.5, 0, .5, 1), 1)
        model = MamdaniFuzzy()
        self.assertLess(model.predict(obs(0)), .3)
        self.assertGreater(model.predict(obs(1)), .7)
        self.assertEqual(model.predict(obs(None)), .5)
        for value in [i / 20 for i in range(21)]:
            self.assertTrue(0 <= model.predict(obs(value)) <= 1)

    def test_ds_hand_calculation(self):
        result = dempster(Mass(.6, .1, .3), Mass(.2, .7, .1))
        self.assertAlmostEqual(result.conflict, .44)
        self.assertAlmostEqual(result.mass.review, .24 / .56)
        self.assertAlmostEqual(result.mass.normal, .29 / .56)
        self.assertAlmostEqual(result.mass.ignorance, .03 / .56)
        self.assertLessEqual(result.mass.interval[0], result.mass.pignistic)
        self.assertGreaterEqual(result.mass.interval[1], result.mass.pignistic)

    def test_ds_vacuous_discount_and_conflict(self):
        m = Mass(.6, .1, .3)
        neutral = dempster(m, Mass(0, 0, 1)).mass
        self.assertAlmostEqual(neutral.review, m.review)
        self.assertEqual(m.discount(0), Mass(0, 0, 1))
        with self.assertRaises(TotalConflict):
            dempster(Mass(1, 0, 0), Mass(0, 1, 0))
        self.assertEqual(yager(Mass(1, 0, 0), Mass(0, 1, 0)).mass, Mass(0, 0, 1))
        self.assertEqual(fuse([Mass(1, 0, 0), Mass(0, 1, 0)]).mass, Mass(0, 0, 1))
        with self.assertRaises(ValueError):
            Mass(.6, .6, 0)

    def test_rule_baseline_uses_shared_features(self):
        model = ScreeningRules()
        self.assertEqual(model.predict(obs(0)), 0)
        self.assertEqual(model.predict(obs(1)), 1)
        self.assertEqual(model.predict(obs(None)), .5)
        self.assertTrue(model.explain(obs(1))["trace"])

    def test_hybrid_preserves_structural_facts(self):
        model = HybridModel()
        locked = [("Alert", "c", "担保圈"), ("Controls", "p", "c")]
        result = model.explain(obs(0), established_facts=locked)
        self.assertEqual(result["locked_facts"], locked)
        self.assertEqual(result["review_priority"], "高")
        self.assertLess(result["score"], .5)
        self.assertTrue(result["structural_alert_requires_review"])
        model.fit_weights([obs(0), obs(1)], [0, 1])
        self.assertEqual(model.validation_count, 2)
        self.assertAlmostEqual(sum(model.weights), 1)
        self.assertEqual(model.predict(obs(.2)), HybridModel.from_dict(model.as_dict()).predict(obs(.2)))
        self.assertEqual(model.without("rule").weights[2], 0)

    def test_feature_mapping_not_all_trade_is_related(self):
        case = load_from_equity_kg()
        fw = ForwardEngine(build_rules(), build_facts(case).to_tuples())
        fw.run()
        values = {o.company_id: o for o in from_case(case, fw)}
        self.assertAlmostEqual(values["C3"].values["related_trade"], 5000 / 80000)
        self.assertEqual(values["C12"].values["related_trade"], 0)
        self.assertEqual(values["C7"].values["cycle"], 1)

    def test_hybrid_respects_medium_rule_priority(self):
        result = HybridModel().explain(obs(0), established_facts=[("RiskLevel", "c", "中")])
        self.assertEqual(result["review_priority"], "中")
        self.assertEqual(result["rule_priority_floor"], "中")

    def test_zero_weight_components_are_not_evaluated(self):
        class Disabled:
            def predict(self, observation):
                raise AssertionError("zero-weight component was evaluated")
        model = HybridModel(fuzzy=Disabled(), weights=(1, 0, 0))
        self.assertEqual(model.predict(obs(.2)), model.bayes.predict(obs(.2)))

    def test_ablation_never_silently_activates_another_component(self):
        for name in HybridModel.NAMES:
            model = HybridModel(weights=tuple(float(n == name) for n in HybridModel.NAMES))
            with self.subTest(component=name), self.assertRaisesRegex(ValueError, "没有剩余正权重"):
                model.without(name)
        model = HybridModel(weights=(1, 0, 0))
        unchanged = model.without("rule")
        self.assertEqual(unchanged.weights, model.weights)
        self.assertEqual(unchanged.ablation["kind"], "unchanged_zero_weight")
        remaining = HybridModel().without("bayes")
        for actual, expected in zip(remaining.weights, (0., .625, .375)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(remaining.ablation["kind"], "renormalized")
        self.assertIsNone(remaining.ablation["fallback_component"])

    def test_explicit_ablation_fallback_is_labeled_and_survives_reload(self):
        original = HybridModel(weights=(1, 0, 0))
        model = original.without("bayes", fallback="fuzzy")
        self.assertEqual(model.weights, (0., 1., 0.))
        self.assertEqual(model.predict(obs(.2)), original.fuzzy.predict(obs(.2)))
        self.assertEqual(model.ablation["kind"], "baseline_fallback")
        self.assertEqual(model.ablation["fallback_component"], "fuzzy")
        self.assertEqual(model.ablation["base_weights"], {"bayes": 1., "fuzzy": 0., "rule": 0.})
        self.assertEqual(model.explain(obs(.2))["ablation"], model.ablation)
        restored = HybridModel.from_dict(model.as_dict())
        self.assertEqual(restored.as_dict(), model.as_dict())
        self.assertEqual(restored.predict(obs(.2)), model.predict(obs(.2)))
        self.assertEqual(original.weights, (1., 0., 0.))
        self.assertIsNone(original.ablation)
        for fallback in ("bayes", "unknown"):
            with self.assertRaises(ValueError):
                original.without("bayes", fallback=fallback)

    def test_bad_observation_rejected(self):
        for value in [-.1, 1.1, float("nan"), float("inf"), True]:
            with self.assertRaises(ValueError):
                obs(value)


if __name__ == "__main__":
    unittest.main()
