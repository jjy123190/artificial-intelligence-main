import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
from core.kb.domain import Holding, load_from_equity_kg, demo_case, SIBLING_DATA, SNAPSHOT_DATA
from core.kb.features import FEATURES, RiskObservation
from core.kb.facts import build_facts
from experiments.lab02.hybrid.model import HybridModel
from experiments.lab02.eval.experiments import ExperimentConfig, MODEL_NAMES, SCENARIOS, run_experiments
from experiments.lab02.eval.metrics import evaluate, roc_auc
from experiments.lab02.eval.synthetic import generate, perturb, assert_disjoint

ROOT = Path(__file__).resolve().parents[2]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="equity_reasoning_tests_")
        cls.base = Path(cls.temp.name).resolve()
        cls.temp_root = Path(tempfile.gettempdir()).resolve()
        cls._verify_temp_path()
        cls.output = cls.base / "experiments"
        cls.results = run_experiments(cls.output, ExperimentConfig((17,), 24, 12, 16), progress=lambda _: None)

    @classmethod
    def _verify_temp_path(cls):
        # 清理前再次核对绝对路径，删除范围只限本测试创建的唯一临时目录。
        if cls.base.parent != cls.temp_root or not cls.base.name.startswith("equity_reasoning_tests_"):
            raise AssertionError("临时目录越界，拒绝清理")

    @classmethod
    def tearDownClass(cls):
        cls._verify_temp_path()
        cls.temp.cleanup()

    def test_data_split_and_repeatability(self):
        a, b, c = [generate(10, 23, split) for split in ("train", "validation", "test")]
        assert_disjoint(a, b, c)
        self.assertEqual(a, generate(10, 23, "train"))
        with self.assertRaises(ValueError):
            assert_disjoint(a, a)
        noisy = perturb(c, sigma=.3, missing=.4, seed=9)
        self.assertEqual(noisy, perturb(c, sigma=.3, missing=.4, seed=9))
        self.assertEqual([s.label for s in c], [s.label for s in noisy])
        self.assertTrue(all(s.observation.coverage == 0 for s in perturb(c, missing=1, seed=9)))

    def test_metrics_known_values(self):
        m = evaluate([0, 1, 0, 1], [.1, .8, .2, .6])
        self.assertEqual(m["accuracy"], 1)
        self.assertEqual(m["f1"], 1)
        self.assertAlmostEqual(m["brier"], .0625)
        self.assertEqual(m["auc"], 1)
        self.assertEqual(roc_auc([0, 1], [.5, .5]), .5)
        self.assertIsNone(roc_auc([1, 1], [.1, .9]))
        with self.assertRaises(ValueError):
            evaluate([], [])

    def test_saved_predictions_reproduce_metrics(self):
        def read(name):
            with (self.output / name).open(encoding="utf-8-sig", newline="") as fh:
                return list(csv.DictReader(fh))
        rows = read("metrics.csv")
        self.assertEqual(len(rows), len(MODEL_NAMES) * len(SCENARIOS))
        predicted = read("predictions.csv")
        for row in rows:
            selected = [p for p in predicted if p["scenario"] == row["scenario"]]
            scores = [float(p[row["model"]]) for p in selected]
            labels = [int(p["label"]) for p in selected]
            accuracy = sum((p >= .5) == bool(y) for p, y in zip(scores, labels)) / len(labels)
            brier = sum((p - y) ** 2 for p, y in zip(scores, labels)) / len(labels)
            self.assertAlmostEqual(float(row["accuracy"]), accuracy)
            self.assertAlmostEqual(float(row["brier"]), brier)
        self.assertEqual(len(read("sensitivity.csv")), 15)
        self.assertEqual(len(read("conflict_metrics.csv")), 24)

    def test_model_reload_matches_saved_prediction(self):
        models = json.loads((self.output / "models.json").read_text(encoding="utf-8"))
        params = models["models_by_seed"]["17"]["model"]
        self.assertEqual(params["validation_count"], 12)
        self.assertEqual(params["bayes"]["training_count"], 24)
        model = HybridModel.from_dict(params)
        with (self.output / "samples.csv").open(encoding="utf-8-sig", newline="") as fh:
            row = next(r for r in csv.DictReader(fh) if r["split"] == "test" and r["scenario"] == "clean")
        obs = RiskObservation(row["sample_id"], {f: float(row[f]) for f in FEATURES})
        with (self.output / "predictions.csv").open(encoding="utf-8-sig", newline="") as fh:
            prediction = next(r for r in csv.DictReader(fh) if r["sample_id"] == row["sample_id"] and r["scenario"] == "clean")
        self.assertAlmostEqual(model.predict(obs), float(prediction["hybrid_tuned"]))

    def test_ablation_exports_actual_operation_and_fallback_counts(self):
        bundle = json.loads((self.output / "models.json").read_text(encoding="utf-8"))["models_by_seed"]["17"]
        with (self.output / "ablations.csv").open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["model"] for r in rows}, set(bundle["ablations"]))
        for row in rows:
            info = bundle["ablations"][row["model"]]
            remaining = sum(w for name, w in info["base_weights"].items() if name != info["removed_component"])
            fallback = remaining == 0
            self.assertEqual(info["kind"] == "baseline_fallback", fallback)
            self.assertEqual(row["fallback_component"], (info["fallback_component"] or ""))
            self.assertEqual(info["result_weights"][info["removed_component"]], 0)
            for summary in self.results["summary"]:
                if summary["model"] == row["model"]:
                    self.assertEqual(summary["ablation_runs"], 1)
                    self.assertEqual(summary["ablation_fallback_runs"], int(fallback))

    def test_source_snapshot_is_exact_copy(self):
        origins = json.loads((ROOT / "data" / "source_hashes.json").read_text(encoding="utf-8"))["files"]
        self.assertEqual(set(origins), {p.name for p in SNAPSHOT_DATA.glob("*.csv")})
        for name, digest in origins.items():
            self.assertEqual(hashlib.sha256((SNAPSHOT_DATA / name).read_bytes()).hexdigest(), digest)

    def test_default_data_is_local_snapshot(self):
        case = load_from_equity_kg()
        self.assertIn(str(SNAPSHOT_DATA.resolve()), case.source)
        self.assertNotIn(str(SIBLING_DATA.resolve()), case.source)
        self.assertEqual(len(case.entities), 16)

    def test_runtime_does_not_depend_on_first_archive(self):
        missing_archive = self.base / "unavailable_first_archive"
        with patch("core.kb.domain.ARCHIVE_DATA", missing_archive), patch("core.kb.domain.SIBLING_DATA", missing_archive):
            self.assertEqual(len(load_from_equity_kg().entities), 16)
            self.assertEqual(len(demo_case().entities), 29)

    def test_missing_files_are_not_silent_empty_data(self):
        d = self.base / "missing_input"
        d.mkdir(exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            load_from_equity_kg(d)

    def test_invalid_data_rejected(self):
        c = load_from_equity_kg()
        c.holdings[0] = replace(c.holdings[0], ratio=float("nan"))
        with self.assertRaises(ValueError):
            c.validate()
        c = load_from_equity_kg()
        c.holdings.append(Holding("P2", "C1", .3))
        with self.assertRaises(ValueError):
            c.validate()
        c = demo_case()
        self.assertLessEqual(sum(h.ratio for h in c.holdings if h.target == "C16"), 1 + 1e-9)

    def test_symmetric_facts_and_structural_preprocessing(self):
        facts = build_facts(load_from_equity_kg()).to_tuples()
        self.assertIn(("Kin", "P2", "P1", "配偶"), facts)
        self.assertIn(("ElevatedReach", "P1", "C3"), facts)

    def test_cli_from_other_working_directory(self):
        output = self.base / "cli_predictions"
        result = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "main.py"), "predict", "--input",
             str(ROOT / "data" / "example_observations.json"), "--output-dir", str(output)],
             cwd=self.base, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads((output / "predictions.json").read_text(encoding="utf-8"))["results"]), 3)

    def test_cli_package_entry_and_invalid_input(self):
        output = self.base / "package_cli"
        result = subprocess.run([sys.executable, "-X", "utf8", "-m", f"{ROOT.name}.main", "query", "Controls", "P3", "C3",
             "--case", "base", "--output-dir", str(output)], cwd=ROOT.parent,
             capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((output / "query.json").read_text(encoding="utf-8"))["answers"], [])
        path = self.base / "empty.json"
        path.write_text("[]", encoding="utf-8")
        result = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "main.py"), "predict", "--input", str(path)],
             cwd=self.base, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("错误", result.stderr)


if __name__ == "__main__":
    unittest.main()
