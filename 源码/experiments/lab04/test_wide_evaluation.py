from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from .evaluate_wide import run


class WideEvaluationBoundaryTests(unittest.TestCase):
    def test_unfinished_training_cannot_open_test_images(self):
        with tempfile.TemporaryDirectory(prefix="lab04_wide_boundary_") as directory:
            root=Path(directory).resolve()
            self.assertEqual(root.parent,Path(tempfile.gettempdir()).resolve())
            candidate=root/"candidate";candidate.mkdir()
            (candidate/"training.json").write_text(json.dumps({"status":"running"}),encoding="utf-8")
            args=SimpleNamespace(candidate=str(candidate),artifacts=str(root/"core"),output=str(root/"out"),data_root="never-open",threads=1)
            with patch("experiments.lab04.evaluate_wide.load_split") as load:
                with self.assertRaisesRegex(ValueError,"Training incomplete"):
                    run(args)
                load.assert_not_called()

    def test_missing_selection_lock_cannot_open_test_images(self):
        with tempfile.TemporaryDirectory(prefix="lab04_wide_boundary_") as directory:
            root=Path(directory).resolve();candidate=root/"candidate";candidate.mkdir()
            (candidate/"training.json").write_text(json.dumps({"status":"complete"}),encoding="utf-8")
            (candidate/"selection_lock.json").write_text(json.dumps({"status":"selecting","test_accessed":False}),encoding="utf-8")
            args=SimpleNamespace(candidate=str(candidate),artifacts=str(root/"core"),output=str(root/"out"),data_root="never-open",threads=1)
            with patch("experiments.lab04.evaluate_wide.load_split") as load:
                with self.assertRaisesRegex(ValueError,"selection lock"):
                    run(args)
                load.assert_not_called()

if __name__=="__main__":unittest.main()
