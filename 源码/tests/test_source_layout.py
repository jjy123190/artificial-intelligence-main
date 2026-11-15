"""Source layout/export and clean lab-three startup must not require old materials."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import patch

from scripts.export_source import ROOT, export, source_files
from scripts.check_source import inspect_source
from scripts import prepare_lab03
from experiments.lab03 import main as lab03_main, service


class SourceExportTests(TestCase):
    def test_export_includes_all_labs_inputs_and_dependencies_but_no_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "source"
            result = export(target)
            self.assertTrue(result["all_three_labs"])
            self.assertFalse(result["generated_artifacts_included"])
            for name in ("experiments/lab01/service.py", "experiments/lab02/service.py",
                         "experiments/lab03/service.py", "requirements-lab03.txt", "data/lab03/benchmark.json",
                         "web/package-lock.json", "scripts/prepare_lab03.py", "scripts/check_source.py"):
                self.assertTrue((target / name).is_file(), name)
            self.assertEqual((target / "打开网站.cmd").read_bytes(), (ROOT / "打开网站.cmd").read_bytes())
            self.assertEqual(inspect_source(target)["status"], "passed")
            self.assertFalse((target / ".git").exists())

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            marker = path / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export(path)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_export_inside_source_is_rejected(self):
        with self.assertRaises(ValueError):
            export(ROOT / "source-export")

    def test_clean_checker_rejects_materials_results_and_installed_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "source"
            export(target)
            for name in ("deliverables/lab01/report.docx", "outputs/lab03/summary.json",
                         "web/node_modules/package/index.js", "web/dist/index.html", "stray.mp4"):
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test fixture", encoding="utf-8")
            result = inspect_source(target)
            self.assertEqual(result["status"], "failed")
            self.assertIn("stray.mp4", result["unexpected_files"])
            self.assertIn("outputs", result["generated_directories"])

    def test_export_file_set_has_no_generated_archive_or_material_extensions(self):
        files = list(source_files())
        forbidden = {".docx", ".pptx", ".pdf", ".mp4", ".mp3", ".png", ".zip", ".npz"}
        self.assertTrue(files)
        self.assertFalse(any(path.suffix.lower() in forbidden for path in files))
        self.assertFalse(any("retained_tools" in path.parts or "dist" in path.parts for path in files))


class CleanStartupTests(TestCase):
    def test_no_results_snapshot_is_honest_and_baselines_still_run(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(service, "OUT", Path(directory)):
            snapshot = service.snapshot()
            self.assertEqual(snapshot["summary"], [])
            self.assertEqual(snapshot["ga"], {})
            self.assertTrue(snapshot["warnings"])
            self.assertTrue(snapshot["queries"])
            for method in ("bfs", "astar_admissible", "astar_fixed", "mcts"):
                result = service.run_task({"query_id": "shared-3", "method": method, "budget": 32})
                self.assertTrue(result["result"]["valid"])
            for method in ("astar_ga", "astar_ml", "hybrid", "hybrid_llm"):
                with self.assertRaises(ValueError):
                    service.run_task({"method": method})

    def test_demo_defaults_to_explicit_untrained_fixed_a_star(self):
        with patch.object(service, "run_task", return_value={"result": {"valid": True}}) as run, \
             patch("builtins.print"):
            self.assertEqual(lab03_main.main(["demo"]), 0)
            self.assertEqual(run.call_args.args[0]["method"], "astar_fixed")

    def test_llm_command_without_history_does_not_require_old_zip_or_make_report(self):
        with patch("pathlib.Path.is_file", return_value=False), \
             patch("experiments.lab03.evaluate.run_llm") as inference, \
             patch("experiments.lab03.reliability.rebuild") as historical, \
             patch("experiments.lab03.report.build_report") as report:
            self.assertEqual(lab03_main.main(["llm"]), 0)
            inference.assert_called_once()
            historical.assert_not_called()
            report.assert_not_called()

    def test_prepare_check_never_creates_results_or_calls_model(self):
        with patch.object(prepare_lab03, "results_present", return_value=False), \
             patch("experiments.lab03.evaluate.run_benchmark") as benchmark, \
             patch("experiments.lab03.evaluate.run_llm") as model:
            self.assertEqual(prepare_lab03.main(["--check"]), 1)
            benchmark.assert_not_called()
            model.assert_not_called()

    def test_prepare_default_runs_baseline_without_report_llm_or_historical_rebuild(self):
        with patch.object(prepare_lab03, "results_present", side_effect=[False, True]), \
             patch("experiments.lab03.evaluate.run_benchmark") as benchmark, \
             patch("experiments.lab03.evaluate.run_llm") as model, \
             patch("experiments.lab03.reliability.rebuild") as historical, \
             patch("experiments.lab03.report.build_report") as report, patch("builtins.print"):
            self.assertEqual(prepare_lab03.main([]), 0)
            benchmark.assert_called_once()
            model.assert_not_called()
            historical.assert_not_called()
            report.assert_not_called()

    def test_missing_or_corrupt_setup_metadata_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertFalse(prepare_lab03.results_present(output))
            for name in prepare_lab03.REQUIRED_FILES:
                (output / name).write_text("{}", encoding="utf-8")
            self.assertFalse(prepare_lab03.results_present(output))
