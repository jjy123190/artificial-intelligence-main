"""完整编程验收的成功/失败路径；正式结果不被测试覆盖。"""
import json
import io
import importlib.util
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from experiments.lab02.eval.verification import verify_results, VerificationError, _compare
from experiments.lab02.eval.experiments import run_experiments, ExperimentConfig
from experiments.lab02.eval.artifacts import seal_manifest
from experiments.lab02.eval.end_to_end import run_end_to_end, EndToEndConfig
from experiments.lab02.eval.benchmark import run_benchmarks
from experiments.lab02.eval.governance_experiments import run_governance_experiments, make_governance_figure
from experiments.lab02.main import run_demo

ROOT = Path(__file__).resolve().parents[2]


class VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="equity_acceptance_tests_")
        cls.base = Path(cls.temp.name).resolve()
        cls.verify_temp()
        with redirect_stdout(io.StringIO()):
            run_demo(cls.base, case_name="base")
        run_experiments(cls.base / "experiments", ExperimentConfig((17,), 12, 8, 8), progress=lambda _: None)
        run_end_to_end(cls.base / "end_to_end", EndToEndConfig((17,), 12, 8, 8), progress=lambda _: None)
        run_benchmarks(cls.base / "benchmarks", groups=(1,), repeats=1, progress=lambda _: None)
        run_governance_experiments(cls.base / "governance", sizes=(4, 8), repeats=1, progress=lambda _: None)

    @classmethod
    def verify_temp(cls):
        if cls.base.parent != Path(tempfile.gettempdir()).resolve() or not cls.base.name.startswith("equity_acceptance_tests_"):
            raise AssertionError("测试临时目录越界")

    @classmethod
    def tearDownClass(cls):
        cls.verify_temp()
        cls.temp.cleanup()

    def test_complete_artifacts_pass_acceptance(self):
        result = verify_results(self.base, ROOT)
        self.assertTrue(result["successful"])
        self.assertTrue(result["source_and_artifact_hashes_match"])
        self.assertEqual(result["indicator_scores_replayed"], 8 * 14)
        self.assertEqual(result["graph_decisions_replayed"], 6 * 11)
        self.assertEqual(result["DS_scores_replayed"], 6 * 4)
        self.assertEqual(result["hard_lock_pairs"], 8 * 6 * 3)

    def test_governance_inputs_are_replayed_without_trusting_saved_claims(self):
        result = verify_results(self.base, ROOT)
        self.assertEqual(result["governance_cases_replayed"], 6)

    def test_full_replay_includes_all_predictions(self):
        result = verify_results(self.base, ROOT, full_replay=True)
        self.assertEqual(result["indicator_scores_replayed"], result["indicator_predictions"] * 14)
        self.assertEqual(result["graph_decisions_replayed"], result["graph_predictions"])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional plotting dependency unavailable")
    def test_all_new_models_can_be_plotted(self):
        from experiments.lab02.eval.plots import make_figures
        from experiments.lab02.eval.e2e_plots import make_e2e_figures
        files = make_figures(self.base / "experiments", self.base / "benchmarks", self.base / "figures")
        files += make_e2e_figures(self.base / "end_to_end", self.base / "end_to_end/figures")
        files.append(make_governance_figure(self.base / "governance", self.base / "governance/figures"))
        self.assertEqual(len(files), 13)
        self.assertTrue(all(p.is_file() and p.stat().st_size > 1000 for p in files))
        self.assertIn("08_adaptive_fuzzy.png", {p.name for p in files})
        self.assertIn("04_calibrated_policy.png", {p.name for p in files})

    def test_missing_metric_field_rejected(self):
        with self.assertRaisesRegex(VerificationError, "缺少结果字段"):
            _compare({"accuracy": .5}, {}, "test")

    def test_tampered_result_rejected_and_file_restored(self):
        target = self.base / "experiments/predictions.csv"
        before = target.read_bytes()
        try:
            target.write_bytes(before + b"tampered")
            with self.assertRaisesRegex(VerificationError, "摘要不一致"):
                verify_results(self.base, ROOT)
        finally:
            target.write_bytes(before)

    def test_resealed_incorrect_ablation_label_is_rejected(self):
        target = self.base / "experiments/models.json"
        manifest_path = self.base / "experiments/manifest.json"
        before, manifest_before = target.read_bytes(), manifest_path.read_bytes()
        try:
            data = json.loads(before)
            info = data["models_by_seed"]["17"]["ablations"]["hybrid_no_bayes"]
            info["kind"] = "renormalized" if info["kind"] != "renormalized" else "baseline_fallback"
            target.write_text(json.dumps(data), encoding="utf-8")
            seal_manifest(target.parent, json.loads(manifest_before))
            with self.assertRaisesRegex(VerificationError, "消融操作说明"):
                verify_results(self.base, ROOT)
        finally:
            target.write_bytes(before)
            manifest_path.write_bytes(manifest_before)

    def test_stale_source_manifest_rejected(self):
        target = self.base / "end_to_end/manifest.json"
        before = target.read_bytes()
        try:
            data = json.loads(before)
            data["source_sha256"]["main.py"] = "0" * 64
            target.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "源码版本已过期"):
                verify_results(self.base, ROOT)
        finally:
            target.write_bytes(before)

    def test_missing_seal_rejected(self):
        target = self.base / "benchmarks/manifest.json"
        before = target.read_bytes()
        try:
            data = json.loads(before)
            del data["artifact_sha256"]
            target.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "缺少结果封存"):
                verify_results(self.base, ROOT)
        finally:
            target.write_bytes(before)

    def test_manifest_path_traversal_rejected(self):
        target = self.base / "demo/manifest.json"
        before = target.read_bytes()
        try:
            data = json.loads(before)
            data["artifact_sha256"]["../../outside.txt"] = "0" * 64
            target.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "非法或缺失结果路径"):
                verify_results(self.base, ROOT)
        finally:
            target.write_bytes(before)
