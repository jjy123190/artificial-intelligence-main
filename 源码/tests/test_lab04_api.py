"""Web integration boundaries independent of optional ML packages and weights."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient
from backend.api import create_app
from backend.lab_four import LabFourService


class LabFourApiTests(unittest.TestCase):
    def setUp(self):
        self.service = Mock()
        self.service.snapshot.return_value = {"available": False}
        self.service.run.return_value = {"compute": "live", "seed": 123}
        self.client = TestClient(create_app(lab_four_service=self.service))

    def test_invalid_episode_choices_cannot_reach_model(self):
        for payload in ({"shots": 2}, {"shots": True}, {"seed": -1}, {"seed": 1.5},
                        {"method": "unknown"}, {"calibration": "unknown"},
                        {"query_labels": [0, 1]}, {"artifact_path": "../outside"}):
            with self.subTest(payload=payload):
                response = self.client.post("/api/labs/lab04/run", json=payload)
                self.assertEqual(response.status_code, 422)
        self.service.run.assert_not_called()

    def test_valid_choices_are_forwarded_without_training_labels(self):
        response = self.client.post("/api/labs/lab04/run", json={"method": "maml", "shots": 5, "seed": 123})
        self.assertEqual(response.status_code, 200)
        self.service.run.assert_called_once_with({"method": "maml", "shots": 5, "seed": 123, "calibration": "full"})

    def test_missing_artifacts_do_not_break_existing_labs(self):
        with tempfile.TemporaryDirectory() as directory:
            client = TestClient(create_app(lab_four_service=LabFourService(Path(directory))))
            self.assertEqual(client.get("/api/health").status_code, 200)
            self.assertEqual(client.get("/api/course").status_code, 200)
            self.assertFalse(client.get("/api/labs/lab04/snapshot").json()["available"])
            self.assertEqual(client.post("/api/labs/lab04/run", json={}).status_code, 503)

    def test_busy_model_does_not_interleave_maml_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            service = LabFourService(Path(directory))
            service._slot.acquire()
            try:
                client = TestClient(create_app(lab_four_service=service))
                self.assertEqual(client.post("/api/labs/lab04/run", json={}).status_code, 429)
            finally:
                service._slot.release()


class LabFourSnapshotEvidenceTests(unittest.TestCase):
    def test_snapshot_keeps_paired_differences_and_revision(self):
        directory = Path(__file__).resolve().parents[1] / "outputs" / "lab04"
        service = LabFourService(directory)
        snapshot = service.snapshot()
        measured = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(snapshot["provenance"]["artifact_revision"]), 64)
        for row in snapshot["comparisons"]:
            entries = measured["rows"]
            if row["method"] in ("graph", "multiview", "wide", "wideview", "continued"):
                entries = json.loads((directory / f"{row['method']}_results.json").read_text(encoding="utf-8"))["rows"]
            source = next(r for r in entries if r["method"] == row["method"] and r["shots"] == row["shot"])
            self.assertEqual(row["paired_delta"], source["paired_delta"])
            self.assertEqual(row["paired_ci95"], source["paired_ci95"])
            self.assertEqual(row["episodes"], source["episodes"])

    def test_invalid_metric_is_rejected_not_rendered_as_zero(self):
        directory = Path(__file__).resolve().parents[1] / "outputs" / "lab04"
        measured = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        measured["rows"][0]["accuracy"] = float("nan")
        from backend.service import WebError
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "results.json").write_text(json.dumps(measured), encoding="utf-8")
            with self.assertRaises(WebError):
                LabFourService(Path(temp)).snapshot()


    def test_web_projection_is_finite_and_posterior_to_prediction(self):
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            self.skipTest("Optional torch unavailable")
        directory = Path(__file__).resolve().parents[1] / "outputs" / "lab04"
        service = LabFourService(directory)
        result = service.run({"method":"calibrated","shots":1,"seed":20261124,"calibration":"graph"})
        self.assertTrue(result["projection"]["computed_after_prediction"])
        self.assertTrue(result["projection"]["isotropic"])
        self.assertEqual(result["projection"]["feature_dimensions"],64)
        self.assertTrue(0 <= sum(result["projection"]["explained_variance"]) <= 1.000001)
        for point in result["points"]:
            self.assertEqual(point["before"],point["after"])


if __name__ == "__main__":
    unittest.main()
