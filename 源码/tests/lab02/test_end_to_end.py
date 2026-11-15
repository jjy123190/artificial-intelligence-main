import csv
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from core.kb.domain import load_from_equity_kg, demo_case
from core.kb.graph_io import case_to_dict, case_from_dict
from core.kb.features import from_case
from experiments.lab02.hybrid.decision import review_decision
from experiments.lab02.hybrid.model import HybridModel
from experiments.lab02.hybrid.pipeline import prepare_case, assess_company
from experiments.lab02.uncertain.bayes import RiskBayes
from experiments.lab02.eval.graph_data import (GraphCondition, generate_graphs, perturb_graph,
                            assert_graph_splits_disjoint, graph_has_cycle_from)
from experiments.lab02.eval.end_to_end import EndToEndConfig, run_end_to_end, E2E_MODELS, decision_metrics

ROOT = Path(__file__).resolve().parents[2]


class GraphPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = generate_graphs(200, 17, "test")
        cls.base = prepare_case(load_from_equity_kg())

    def test_graph_json_roundtrip_and_unknown_fields(self):
        original = self.samples[0].case
        self.assertEqual(case_from_dict(case_to_dict(original)), original)
        bad = case_to_dict(original)
        bad["audit_label"] = 1
        with self.assertRaises(ValueError):
            case_from_dict(bad)
        bad = case_to_dict(original)
        bad["holdings"][0]["ratio"] = float("nan")
        with self.assertRaises(ValueError):
            case_from_dict(bad)

    def test_generator_repeatable_and_labels_not_rule_outputs(self):
        self.assertEqual(generate_graphs(3, 17, "test"), self.samples[:3])
        self.assertTrue(any(s.label == 0 and s.oracle_alerts for s in self.samples))
        self.assertTrue(any(s.label == 1 and not s.oracle_alerts for s in self.samples))
        assert_graph_splits_disjoint(generate_graphs(5, 17, "train"), generate_graphs(5, 17, "validation"), self.samples)
        with self.assertRaises(ValueError):
            assert_graph_splits_disjoint(self.samples, self.samples)

    def test_oracle_structures_match_independent_traversal_and_rules(self):
        for sample in self.samples[:16]:
            case = sample.case
            h = graph_has_cycle_from([(e.holder, e.target) for e in case.holdings], sample.target_id)
            g = graph_has_cycle_from([(e.guarantor, e.beneficiary) for e in case.guarantees], sample.target_id)
            self.assertEqual(h, "循环持股" in sample.oracle_alerts)
            self.assertEqual(g, "担保圈" in sample.oracle_alerts)
            context = prepare_case(case)
            alerts = tuple(sorted(f[2] for f in context.established_for(sample.target_id) if f[0] == "Alert"))
            self.assertEqual(alerts, sample.oracle_alerts)

    def test_policy_does_not_change_score_or_base_class(self):
        facts = [("Alert", "c", "担保圈")]
        a = review_decision(.1, "c", established_facts=facts, lock_structure=False)
        b = review_decision(.1, "c", established_facts=facts, lock_structure=True)
        self.assertEqual(a["score"], b["score"])
        self.assertEqual(a["risk_class"], b["risk_class"])
        self.assertFalse(a["review_required"])
        self.assertTrue(b["review_required"])
        self.assertTrue(a["constraint_violation"])
        self.assertFalse(b["constraint_violation"])
        self.assertTrue(b["lock_changed_action"])
        self.assertEqual(a["locked_facts"], [])

    def test_threshold_and_priority_are_consistent(self):
        self.assertTrue(review_decision(.5, "c")["review_required"])
        self.assertEqual(review_decision(.5, "c")["review_priority"], "高")
        facts = [("RiskLevel", "c", "中")]
        self.assertTrue(review_decision(.1, "c", established_facts=facts, lock_structure=False)["priority_floor_violation"])
        self.assertFalse(review_decision(.1, "c", established_facts=facts)["priority_floor_violation"])
        for threshold in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                review_decision(.1, "c", threshold=threshold)
        with self.assertRaises(ValueError):
            review_decision(.1, "c", established_facts=[("Alert", "c")])

    def test_common_pipeline_matches_model_explanation(self):
        model = HybridModel()
        result = assess_company(self.base, "C3", model, include_explanation=True)
        direct = model.explain(self.base.observations["C3"], established_facts=self.base.established_for("C3"))
        for key in ("score", "risk_class", "review_required", "review_priority", "lock_changed_action"):
            self.assertEqual(result[key], direct[key])
        self.assertTrue(result["proofs"])
        self.assertTrue(result["review_required"])

    def test_complete_features_equal_legacy_values(self):
        legacy = {o.company_id: o.values for o in from_case(self.base.case, self.base.engine)}
        self.assertEqual(legacy, {k: v.values for k, v in self.base.observations.items()})

    def test_missing_records_are_unknown_not_zero(self):
        sample = self.samples[0]
        observed = perturb_graph(sample.case, sample.target_id, GraphCondition("all_missing", edge_missing=1), 7)
        context = prepare_case(observed.case, completeness=observed.completeness)
        values = context.observations[sample.target_id].values
        self.assertIsNone(values["cycle"])
        self.assertIsNone(values["related_trade"])
        self.assertIsNone(values["guarantee"])
        self.assertIsNotNone(values["opacity"])
        with self.assertRaises(ValueError):
            prepare_case(sample.case, completeness={"holdings": "false"})

    def test_partial_trade_records_do_not_prove_shell(self):
        case = demo_case()
        complete = prepare_case(case)
        partial = prepare_case(case, completeness={"trades": False})
        self.assertIn(("Shell", "C20"), complete.facts.to_tuples())
        self.assertNotIn(("Shell", "C20"), partial.facts.to_tuples())

    def test_baselines_report_only_active_scoring_components(self):
        result = assess_company(self.base, "C3", method="bayes_only", include_explanation=True)
        self.assertEqual(result["weights"], {"bayes": 1.0})
        self.assertEqual(result["components"], {"bayes": result["score"]})
        result = assess_company(self.base, "C3", method="rules_only", include_explanation=True)
        self.assertEqual(result["weights"], {"structural_rules": 1.0})

    def test_noise_preserves_validity_and_does_not_mutate_reference(self):
        sample = self.samples[0]
        before = case_to_dict(sample.case)
        condition = GraphCondition("stress", .5, .5, .5, .3, 1)
        a = perturb_graph(sample.case, sample.target_id, condition, 29)
        b = perturb_graph(sample.case, sample.target_id, condition, 29)
        self.assertEqual(a.case, b.case)
        a.case.validate()
        self.assertEqual(before, case_to_dict(sample.case))

    def test_false_source_alert_can_cause_extra_false_positive(self):
        sample = next(s for s in self.samples if s.label == 0 and not s.oracle_alerts)
        observed = perturb_graph(sample.case, sample.target_id, GraphCondition("false_loop", spurious_return=1), 9)
        self.assertEqual(observed.corruption_stats["injected_guarantee_returns"], 1)
        context = prepare_case(observed.case)
        model = HybridModel(bayes=RiskBayes(prior=.000001), weights=(1, 0, 0))
        a = assess_company(context, sample.target_id, model, method="hybrid_unlocked")
        b = assess_company(context, sample.target_id, model, method="hybrid_locked")
        self.assertFalse(a["review_required"])
        self.assertTrue(b["review_required"])
        self.assertFalse(sample.oracle_alerts)
        self.assertEqual(sample.label, 0)

    def test_labels_are_not_pipeline_inputs(self):
        sample = self.samples[0]
        other = replace(sample, label=1 - sample.label)
        a = assess_company(prepare_case(sample.case), sample.target_id)
        b = assess_company(prepare_case(other.case), other.target_id)
        for key in ("score", "risk_class", "review_required", "review_priority"):
            self.assertEqual(a[key], b[key])

    def test_rules_only_preserves_middle_priority(self):
        result = assess_company(self.base, "C12", method="rules_only")
        if result["rule_priority_floor"] == "中":
            self.assertEqual(result["review_priority"], "中")
        self.assertFalse(result["priority_floor_violation"])


class EndToEndArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="equity_graph_tests_")
        cls.base = Path(cls.temp.name).resolve()
        cls._verify()
        cls.output = cls.base / "end_to_end"
        cls.results = run_end_to_end(cls.output, EndToEndConfig((17,), 12, 8, 8), progress=lambda _: None)

    @classmethod
    def _verify(cls):
        if cls.base.parent != Path(tempfile.gettempdir()).resolve() or not cls.base.name.startswith("equity_graph_tests_"):
            raise AssertionError("临时测试目录越界，拒绝清理")

    @classmethod
    def tearDownClass(cls):
        cls._verify()
        cls.temp.cleanup()

    def read_csv(self, filename):
        with (self.output / filename).open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def test_metrics_use_actions_and_can_be_recomputed(self):
        predictions = self.read_csv("predictions.csv")
        metrics = self.read_csv("metrics.csv")
        self.assertEqual(len(metrics), len(E2E_MODELS) * 6)
        for row in metrics:
            group = [r for r in predictions if r["scenario"] == row["scenario"] and r["model"] == row["model"]]
            result = decision_metrics(group)
            correct = sum(int(r["review_required"]) == int(r["label"]) for r in group) / len(group)
            self.assertAlmostEqual(correct, float(row["accuracy"]))
            self.assertAlmostEqual(result["score_brier"], float(row["score_brier"]))
            self.assertEqual(result["constraint_violations"], int(row["constraint_violations"]))
        locked = [r for r in predictions if r["model"].endswith("_locked")]
        self.assertTrue(all(int(r["constraint_violation"]) == 0 for r in locked))

    def test_paired_scores_unchanged_and_negative_results_retained(self):
        paired = self.read_csv("paired_effects.csv")
        self.assertEqual(len(paired), 18)
        self.assertEqual({r["family"] for r in paired}, {"fixed", "tuned", "adaptive"})
        for row in paired:
            self.assertEqual(float(row["delta_score_brier"]), 0)
            self.assertGreaterEqual(int(row["delta_fp"]), 0)
            self.assertLessEqual(int(row["delta_fn"]), 0)
        self.assertIn("extra_false_positive", json.loads((self.output / "examples.json").read_text(encoding="utf-8")))

    def test_saved_observed_graph_replays_without_label(self):
        with (self.output / "observed_graphs.jsonl").open(encoding="utf-8") as f:
            raw = json.loads(next(f))
        self.assertNotIn("label", raw)
        self.assertNotIn("oracle_alerts", raw)
        context = prepare_case(case_from_dict(raw["case"]), completeness=raw["completeness"])
        bundle = json.loads((self.output / "models.json").read_text(encoding="utf-8"))["models_by_seed"]["17"]
        model = HybridModel.from_dict(bundle["fixed_model"])
        actual = assess_company(context, raw["target_id"], model)
        expected = next(r for r in self.read_csv("predictions.csv") if r["sample_id"] == raw["sample_id"]
                        and r["scenario"] == raw["scenario"] and r["model"] == "hybrid_fixed_locked")
        self.assertEqual(actual["review_required"], bool(int(expected["review_required"])))
        self.assertAlmostEqual(actual["score"], float(expected["score"]))

    def test_cli_graph_entry_uses_same_pipeline(self):
        sample = generate_graphs(1, 2, "test")[0]
        request = self.base / "graph.json"
        request.write_text(json.dumps({"case": case_to_dict(sample.case), "target_id": sample.target_id}, ensure_ascii=False), encoding="utf-8")
        output = self.base / "cli"
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / "main.py"), "analyze",
            "--input", str(request), "--output-dir", str(output)], cwd=self.base,
            capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads((output / "graph_analysis.json").read_text(encoding="utf-8"))
        expected = assess_company(prepare_case(sample.case), sample.target_id)
        self.assertAlmostEqual(data["decisions"][0]["score"], expected["score"])
        self.assertEqual(data["decisions"][0]["review_required"], expected["review_required"])
        self.assertTrue(data["graph"]["steps"])

    def test_cli_fixed_model_replays_saved_prediction(self):
        with (self.output / "observed_graphs.jsonl").open(encoding="utf-8") as f:
            raw = json.loads(next(f))
        path = self.base / "replay_request.json"
        path.write_text(json.dumps({k: raw[k] for k in ("case", "target_id", "completeness")}), encoding="utf-8")
        output = self.base / "fixed_replay"
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / "main.py"), "analyze",
            "--input", str(path), "--model", str(self.output / "models.json"), "--model-seed", "17",
            "--model-variant", "fixed", "--output-dir", str(output)], cwd=self.base,
            capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads((output / "graph_analysis.json").read_text(encoding="utf-8"))["decisions"][0]
        expected = next(r for r in self.read_csv("predictions.csv") if r["sample_id"] == raw["sample_id"]
                        and r["scenario"] == raw["scenario"] and r["model"] == "hybrid_fixed_locked")
        self.assertAlmostEqual(actual["score"], float(expected["score"]))
        self.assertEqual(actual["review_required"], bool(int(expected["review_required"])))

    def test_train_validation_test_and_models_are_auditable(self):
        audit = json.loads((self.output / "split_audit.json").read_text(encoding="utf-8"))[0]
        self.assertEqual(audit["sample_id_overlap"], 0)
        self.assertEqual(audit["entity_id_overlap"], 0)
        model = json.loads((self.output / "models.json").read_text(encoding="utf-8"))["models_by_seed"]["17"]["model"]
        self.assertEqual(model["bayes"]["training_count"], 12)
        self.assertEqual(model["validation_count"], 8)
        self.assertTrue((self.output / "config.json").exists())


if __name__ == "__main__":
    unittest.main()
