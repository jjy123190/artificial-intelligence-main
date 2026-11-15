"""Optional web tests. The standard-library core test suite needs no web dependencies."""
from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from backend.api import create_app
    from backend.schemas import MAX_BODY_BYTES, AnalyzeRequest
    from backend.service import InferenceService, WebError, run_bounded
    from backend.worker import analyze_payload, query_payload, preset_case
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]


def inline_runner(operation, payload):
    return analyze_payload(payload) if operation == "analyze" else query_payload(payload)


@unittest.skipUnless(WEB_AVAILABLE, "optional requirements-web.txt not installed")
class WebApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = InferenceService(runner=inline_runner, cache_size=20)
        cls.client = TestClient(create_app(cls.service))
        cls.payload = AnalyzeRequest(case=preset_case("base")).model_dump()
        response = cls.client.post("/api/analyze", json=cls.payload)
        if response.status_code != 200:
            raise AssertionError(response.text)
        cls.analysis = response.json()

    def test_bootstrap_presets_rules_experiments(self):
        self.assertEqual(self.client.get("/api/health").json()["status"], "ok")
        self.assertEqual(len(self.client.get("/api/bootstrap").json()["methods"]), 6)
        self.assertEqual(len(self.client.get("/api/rules").json()["items"]), 27)
        self.assertIn("合成", self.client.get("/api/experiments").json()["note"])
        for name in ("base", "extended"):
            case = self.client.get("/api/cases/" + name).json()["case"]
            self.assertGreater(len(case["entities"]), 0)
        self.assertEqual(self.client.get("/api/cases/nope").status_code, 404)

    def test_rule_response_exposes_complete_trigger_conditions(self):
        items = self.client.get("/api/rules").json()["items"]
        self.assertTrue(all(rule["guard_complete"] for rule in items))
        by_name = {rule["name"]: rule for rule in items}
        self.assertEqual(by_name["R01"]["guard_conditions"], ["?r > 0.5", "?x ≠ ?y"])
        self.assertEqual(by_name["R02"]["guard_conditions"], ['?rel = "配偶"'])
        self.assertEqual(by_name["R04"]["guard_conditions"], ["?r1 ≥ ?r2", "?r1 + ?r2 > 0.5", "?p ≠ ?q"])
        self.assertEqual(by_name["R08"]["guard_conditions"], [])

    def test_same_core_and_lock_does_not_modify_score(self):
        from experiments.lab02.hybrid.pipeline import prepare_case, assess_company
        from experiments.lab02.hybrid.model import HybridModel
        from core.kb.graph_io import case_from_dict
        context = prepare_case(case_from_dict(self.payload["case"]))
        direct = assess_company(context, "C3", HybridModel(), method="hybrid_locked")
        shown = self.analysis["decisions"]["C3"]
        self.assertAlmostEqual(shown["score"], direct["score"], places=12)
        pair = shown["hybrid_comparison"]
        self.assertEqual(pair["locked"]["score"], pair["unlocked"]["score"])
        self.assertTrue(pair["locked"]["review_required"])
        self.assertFalse(pair["unlocked"]["review_required"])
        response = self.client.post("/api/analyze", json={**self.payload, "method": "hybrid_unlocked"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["decisions"]["C3"]["review_required"])

    def test_calibrated_method_keeps_provenance_and_raw_score(self):
        response = self.client.post("/api/analyze", json={**self.payload, "method": "hybrid_calibrated"})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["parameters"]["method"], "hybrid_calibrated")
        result = data["decisions"]["C3"]
        self.assertIn("deferred_structural_alert", result)
        self.assertEqual(result["score"], self.analysis["decisions"]["C3"]["score"])
        self.assertEqual(set(data["provenance"]), set(self.analysis["provenance"]))

    def test_actual_holding_edit_changes_control_proof(self):
        changed = copy.deepcopy(self.payload)
        edge = next(h for h in changed["case"]["holdings"] if h["holder"] == "C2" and h["target"] == "C3")
        self.assertEqual(edge["ratio"], .35)
        edge["ratio"] = .25
        response = self.client.post("/api/analyze", json=changed)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertFalse(any(e["source"] == "P1" and e["target"] == "C3" for e in result["derived_edges"]))
        self.assertNotIn("P1", result["decisions"]["C3"]["controllers"])
        self.assertEqual(self.analysis["case"]["holdings"], self.payload["case"]["holdings"])

    def test_backward_queries_true_false_and_variable_rejected(self):
        for source, proved in (("P1", True), ("P3", False)):
            response = self.client.post("/api/query", json={"analysis_id": self.analysis["id"], "predicate": "Controls", "arguments": [source, "C3"]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["proved"], proved)
            self.assertTrue(response.json()["equivalent"])
        # Worker semantics: variables are rejected rather than accidentally treated as query wildcards.
        with self.assertRaises(ValueError):
            query_payload({**self.payload, "predicate": "Controls", "arguments": ["?p", "C3"]})

    def test_provenance_and_step_references_are_complete(self):
        nodes = self.analysis["provenance"]
        for fid, fact in nodes.items():
            self.assertEqual(fid, fact["id"])
            self.assertTrue(fact["source"])
            self.assertTrue(all(parent in nodes for parent in fact["premises"]))
        for step in self.analysis["steps"]:
            self.assertTrue(all(fid in nodes for fid in step["fact_ids"]))
        for result in self.analysis["decisions"].values():
            self.assertTrue(all(fid in nodes for fid in result["claim_ids"]))

    def test_cache_hit_and_saved_analysis(self):
        response = self.client.post("/api/analyze", json=self.payload).json()
        self.assertTrue(response["cached"])
        self.assertEqual(response["id"], self.analysis["id"])
        self.assertEqual(self.client.get("/api/analyses/" + response["id"]).json()["id"], response["id"])
        self.assertEqual(self.client.get("/api/analyses/" + "0" * 32).status_code, 404)

    def test_body_media_and_size_guards(self):
        for content, headers, status in ((b"{", {"content-type": "application/json"}, 400),
                                        (b"{}", {"content-type": "text/plain"}, 415),
                                        (b"{}", {"content-type": "application/json", "content-encoding": "gzip"}, 415),
                                        (b" " * (MAX_BODY_BYTES + 1), {"content-type": "application/json"}, 413)):
            with self.subTest(status=status, headers=headers):
                self.assertEqual(self.client.post("/api/analyze", content=content, headers=headers).status_code, status)

    def test_invalid_parameter_types_and_labels(self):
        for patch_value in ({"threshold": True}, {"threshold": "0.5"}, {"threshold": 2},
                            {"method": "eval"}, {"model_source": "../../auth.json"},
                            {"label": 1}, {"completeness": {"holdings": "false"}}):
            with self.subTest(value=patch_value):
                self.assertEqual(self.client.post("/api/analyze", json={**self.payload, **patch_value}).status_code, 422)
        nan = json.dumps({**self.payload, "threshold": float("nan")})
        self.assertEqual(self.client.post("/api/analyze", content=nan, headers={"content-type": "application/json"}).status_code, 422)

    def test_invalid_graph_references_duplicates_and_total(self):
        invalid = []
        case = copy.deepcopy(self.payload["case"])
        case["holdings"][0]["target"] = "NO_SUCH_COMPANY"
        invalid.append(case)
        case = copy.deepcopy(self.payload["case"])
        case["entities"].append(case["entities"][0])
        invalid.append(case)
        case = copy.deepcopy(self.payload["case"])
        for holding in case["holdings"]:
            if holding["target"] == "C3":
                holding["ratio"] = .9
        invalid.append(case)
        case = copy.deepcopy(self.payload["case"])
        case["entities"] = [e for e in case["entities"] if e["kind"] == "Person"]
        invalid.append(case)
        for case in invalid:
            self.assertEqual(self.client.post("/api/analyze", json={**self.payload, "case": case}).status_code, 422)

    def test_large_cycle_is_rejected_before_inference(self):
        case = {"schema_version": 1, "entities": [{"eid": f"E{i}", "name": f"Enterprise {i}", "kind": "Company"} for i in range(9)],
                "holdings": [{"holder": f"E{i}", "target": f"E{(i+1)%9}", "ratio": .6} for i in range(9)]}
        result = self.client.post("/api/analyze", json={**self.payload, "case": case})
        self.assertEqual(result.status_code, 422)
        self.assertIn("8个实体", result.json()["error"]["message"])

    def test_host_origin_security_headers_and_no_file_access(self):
        self.assertEqual(self.client.get("/api/health", headers={"host": "attacker.example"}).status_code, 400)
        self.assertEqual(self.client.post("/api/analyze", json=self.payload, headers={"origin": "https://attacker.example"}).status_code, 403)
        response = self.client.get("/api/health")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(self.client.get("/api/read?path=README.md").status_code, 404)
        self.assertEqual(self.client.get("/%2e%2e/README.md").status_code, 404)

    def test_case_snapshots_remain_unchanged(self):
        files = list((ROOT / "data" / "raw").rglob("*.csv"))
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        self.client.post("/api/analyze", json=self.payload)
        for p, digest in before.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)


@unittest.skipUnless(WEB_AVAILABLE, "optional requirements-web.txt not installed")
class WebServiceTests(unittest.TestCase):
    def test_expiry_and_duplicate_fingerprint_eviction(self):
        service = InferenceService(runner=lambda *_: {}, cache_size=1)
        payload = {"model_source": "manual"}
        first = service.analyze(payload)
        # Simulate two concurrently completed computations of the same fingerprint.
        record = service._cache[first["id"]]
        service._cache["new"] = {**record, "data": {**record["data"], "id": "new"}}
        service._keys[record["key"]] = "new"
        service._prune()
        self.assertEqual(service.analyze(payload)["id"], "new")
        service._cache["new"]["time"] -= 1900
        with self.assertRaises(WebError) as error:
            service.get("new")
        self.assertEqual(error.exception.code, "analysis_expired")

    def test_busy_slot_returns_retryable_error(self):
        service = InferenceService(runner=lambda *_: {})
        service._slots.acquire()
        service._slots.acquire()
        try:
            with self.assertRaises(WebError) as error:
                service.analyze({"model_source": "manual"})
            self.assertEqual(error.exception.status, 429)
        finally:
            service._slots.release()
            service._slots.release()

    def test_corrupt_optional_model_catalog_falls_back_to_manual(self):
        from backend.worker import model_catalog
        for content in ("[]", '{"version":1,"models_by_seed":[]}',
                        '{"version":1,"models_by_seed":{"7":{"model":null}}}'):
            with patch("pathlib.Path.is_file", return_value=True), patch("pathlib.Path.read_text", return_value=content):
                self.assertEqual([item["id"] for item in model_catalog()], ["manual"])

    def test_real_spawn_and_timeout_leave_no_owned_worker(self):
        payload = AnalyzeRequest(case=preset_case("base")).model_dump()
        before = {p.pid for p in multiprocessing.active_children()}
        result = run_bounded("analyze", payload)
        self.assertTrue(result["decisions"]["C3"]["review_required"])
        with self.assertRaises(WebError) as error:
            run_bounded("analyze", payload, timeout=0)
        self.assertEqual(error.exception.code, "timeout")
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)


if __name__ == "__main__":
    unittest.main()
