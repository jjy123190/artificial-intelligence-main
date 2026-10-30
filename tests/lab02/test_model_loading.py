"""训练结果必须能够被命令行和网页重新加载，且不改变预测。"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend import worker
from core.kb.features import FEATURES, RiskObservation
from experiments.lab02.hybrid.model import HybridModel
from experiments.lab02.main import _hybrid, main
from experiments.lab02.uncertain.fuzzy import MamdaniFuzzy


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lab02_model_loading_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "models.json"
        self.models = {
            "model": HybridModel(weights=(1, 0, 0)),
            "fixed_model": HybridModel(),
            "adaptive_model": HybridModel(fuzzy=MamdaniFuzzy(high_start=.30), weights=(.5, .5, 0)),
        }
        self.observation = RiskObservation("C1", dict(zip(FEATURES, (.2, .8, .3, .7))))

    def write_archive(self, version):
        self.path.write_text(json.dumps({
            "version": version,
            "models_by_seed": {"7": {key: model.as_dict() for key, model in self.models.items()}},
        }), encoding="utf-8")

    def test_cli_and_web_load_all_variants_from_both_versions(self):
        for version in (1, 2):
            self.write_archive(version)
            with patch.dict(worker.MODEL_FILES, {"indicator": self.path}, clear=True):
                ids = {item["id"] for item in worker.model_catalog()}
                for variant, key in (("selected", "model"), ("fixed", "fixed_model"),
                                     ("adaptive", "adaptive_model")):
                    with self.subTest(version=version, variant=variant):
                        identifier = f"indicator:7:{variant}"
                        self.assertIn(identifier, ids)
                        cli_model, _ = _hybrid(self.path, "7", variant)
                        web_model = worker.load_model(identifier)
                        expected = self.models[key].predict(self.observation)
                        self.assertAlmostEqual(cli_model.predict(self.observation), expected)
                        self.assertAlmostEqual(web_model.predict(self.observation), expected)

    def test_predict_command_accepts_current_training_format(self):
        self.write_archive(2)
        inputs = self.root / "observations.json"
        inputs.write_text(json.dumps([self.observation.as_dict()]), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            status = main(["predict", "--input", str(inputs), "--model", str(self.path),
                           "--output-dir", str(self.root / "predictions")])
        self.assertEqual(status, 0)
        results = json.loads((self.root / "predictions/predictions.json").read_text(encoding="utf-8"))
        self.assertEqual(results["results"][0]["company_id"], "C1")

    def test_unknown_versions_are_rejected(self):
        for version in (0, 3, True, "2"):
            self.write_archive(version)
            with self.subTest(version=version):
                with self.assertRaises(ValueError):
                    _hybrid(self.path)
                with patch.dict(worker.MODEL_FILES, {"indicator": self.path}, clear=True):
                    self.assertEqual([row["id"] for row in worker.model_catalog()], ["manual"])

    def test_malformed_archives_and_missing_seed_are_reported(self):
        for data in ([], {"version": 2}, {"version": 2, "models_by_seed": []},
                     {"version": 2, "models_by_seed": {"7": []}}):
            self.path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                _hybrid(self.path)
        self.write_archive(2)
        with self.assertRaisesRegex(ValueError, "seed=19"):
            _hybrid(self.path, "19")


if __name__ == "__main__":
    unittest.main()
