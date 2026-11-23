"""Web integration boundaries independent of optional ML packages and weights."""
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


if __name__ == "__main__":
    unittest.main()
