"""自动参数拟合与行动校准回归：预测完整重载、基线、缺测及错误来源。"""
import copy
import unittest
from unittest.mock import patch
from experiments.lab02.uncertain.fuzzy import MamdaniFuzzy
from experiments.lab02.hybrid.model import HybridModel
from experiments.lab02.hybrid.calibration import ReviewPolicy
from experiments.lab02.hybrid.pipeline import prepare_case, assess_company
from experiments.lab02.uncertain.bayes import RiskBayes
from experiments.lab02.eval.synthetic import generate
from core.kb.features import RiskObservation
from core.kb.domain import load_from_equity_kg


class AdaptiveFuzzyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        samples = generate(16, 29, "train")
        cls.obs = [s.observation for s in samples]
        cls.labels = [s.label for s in samples]
        cls.fitted = MamdaniFuzzy.fit(cls.obs, cls.labels)

    def test_training_search_contains_manual_baseline(self):
        self.assertEqual(len(self.fitted.fit_history), 15)
        baseline = next(r for r in self.fitted.fit_history if r["high_start"] == .45 and r["medium_weight"] == 1)
        self.assertLessEqual(self.fitted.training_brier, baseline["training_brier"])
        self.assertEqual(self.fitted.training_count, 16)

    def test_fit_repeatable_and_inputs_unchanged(self):
        before = copy.deepcopy(self.obs)
        self.assertEqual(self.fitted.as_dict(), MamdaniFuzzy.fit(self.obs, self.labels).as_dict())
        self.assertEqual(self.obs, before)

    def test_complete_serialization_preserves_grid_and_parameters(self):
        fuzzy = MamdaniFuzzy(high_start=.6, grid_points=51, medium_weight=.35)
        model = HybridModel(fuzzy=fuzzy, weights=(0, 1, 0))
        model.review_policy = ReviewPolicy(.35)
        reloaded = HybridModel.from_dict(model.as_dict())
        self.assertEqual(reloaded.as_dict(), model.as_dict())
        for obs in self.obs:
            self.assertEqual(reloaded.predict(obs), model.predict(obs))
        self.assertEqual(MamdaniFuzzy.from_dict(self.fitted.as_dict()).as_dict(), self.fitted.as_dict())

    def test_ablation_retains_calibrated_policy(self):
        model = HybridModel(fuzzy=self.fitted)
        model.review_policy = ReviewPolicy(.45)
        self.assertEqual(model.without("rule").review_policy.as_dict(), model.review_policy.as_dict())

    def test_legacy_model_without_fuzzy_bundle_remains_loadable(self):
        model = HybridModel()
        raw = model.as_dict()
        del raw["fuzzy"]
        del raw["review_policy"]
        old = HybridModel.from_dict(raw)
        for obs in self.obs:
            self.assertEqual(old.predict(obs), model.predict(obs))

    def test_missing_is_neutral_and_fit_does_not_happen_at_predict(self):
        obs = RiskObservation("missing", {})
        with patch.object(MamdaniFuzzy, "fit", side_effect=AssertionError("test-time training")):
            self.assertEqual(self.fitted.predict(obs), .5)
        self.assertEqual(self.fitted.explain(obs)["coverage"], 0)

    def test_invalid_fit_and_parameters_rejected(self):
        for kwargs in ({"grid_points": 11.5}, {"grid_points": True}, {"high_start": float("nan")},
                       {"medium_weight": 0}, {"medium_weight": float("inf")}):
            with self.assertRaises(ValueError):
                MamdaniFuzzy(**kwargs)
        for obs, labels in (([], []), (self.obs, [0]), (self.obs, [True] * len(self.obs)),
                            (self.obs, [2] * len(self.obs))):
            with self.assertRaises(ValueError):
                MamdaniFuzzy.fit(obs, labels)


class ReviewPolicyTests(unittest.TestCase):
    def test_gate_retains_proofs_not_false_probability(self):
        facts = [("Alert", "C3", "循环持股")]
        decision = ReviewPolicy(.35).decide(.1, "C3", established_facts=facts)
        self.assertEqual(decision["score"], .1)
        self.assertFalse(decision["review_required"])
        self.assertTrue(decision["deferred_structural_alert"])
        self.assertTrue(decision["constraint_violation"])
        self.assertEqual(decision["retained_proofs"], facts)
        supported = ReviewPolicy(.35).decide(.4, "C3", established_facts=facts)
        self.assertTrue(supported["review_required"])
        self.assertTrue(supported["lock_changed_action"])
        self.assertEqual(supported["score"], .4)

    def test_gate_does_not_create_alert_for_other_company(self):
        result = ReviewPolicy(0).decide(.1, "C3", established_facts=[("Alert", "C2", "循环持股")])
        self.assertFalse(result["review_required"])
        self.assertFalse(result["structural_alert_requires_review"])

    def test_validation_search_never_uses_test_or_oracle(self):
        rows = [{"score": .1, "observed_required": True, "label": 0},
                {"score": .4, "observed_required": True, "label": 1}]
        model = ReviewPolicy().fit(rows)
        self.assertEqual(model.validation_count, 2)
        self.assertEqual(len(model.selection_history), 6)
        self.assertEqual(min(r["cost"] for r in model.selection_history), 0)
        self.assertEqual(ReviewPolicy.from_dict(model.as_dict()).as_dict(), model.as_dict())

    def test_invalid_policy_parameters_and_data(self):
        for value in (True, -1, float("nan")):
            with self.assertRaises(ValueError): ReviewPolicy(value)
        for value in (0, float("inf")):
            with self.assertRaises(ValueError): ReviewPolicy(false_negative_cost=value)
        for rows in ([], [{"score": .2, "observed_required": 1, "label": 0}],
                     [{"score": True, "observed_required": True, "label": 1}]):
            with self.assertRaises(ValueError): ReviewPolicy().fit(rows)

    def test_calibrated_pipeline_keeps_hard_and_unlocked_controls(self):
        context = prepare_case(load_from_equity_kg())
        model = HybridModel(bayes=RiskBayes(prior=.000001), weights=(1, 0, 0))
        model.review_policy = ReviewPolicy(.45)
        before = set(context.engine.wm.facts)
        unlocked = assess_company(context, "C3", model, method="hybrid_unlocked")
        hard = assess_company(context, "C3", model, method="hybrid_locked")
        calibrated = assess_company(context, "C3", model, method="hybrid_calibrated")
        self.assertEqual(unlocked["score"], hard["score"])
        self.assertEqual(calibrated["score"], hard["score"])
        self.assertTrue(hard["review_required"])
        self.assertFalse(calibrated["review_required"])
        self.assertEqual(calibrated["retained_proofs"], context.established_for("C3"))
        self.assertEqual(before, set(context.engine.wm.facts))
