"""证据/时态/协议/权益/持久化历史的回归，全部使用隔离临时数据库。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from core.kb.domain import EquityCase, Entity, Holding, load_from_equity_kg
from core.kb.graph_io import case_to_dict
from core.kb.governance import governance_snapshot, claim_changes
from core.kb.equity import economic_equity
from core.kb.history import GovernanceHistory, RevisionConflict


def payload():
    return {"case": case_to_dict(load_from_equity_kg()), "as_of": "2026-10-22", "target": "C3",
            "strict_evidence": False, "strict_identity": False, "documents": [],
            "identities": [], "relations": [], "rights": []}


def document(id_="ev1", end=""):
    return {"id": id_, "title": "教学核验文本", "content": "虚构企业登记/协议核查说明", "sha256": "",
            "status": "verified", "reviewer": "教学核验人", "valid_from": "", "valid_to": end}


def right(party="P3", kind="votes", share=.6, id_="r1", evidence="ev1"):
    return {"id": id_, "party": party, "target": "C3", "kind": kind, "share": share,
            "evidence_id": evidence, "valid_from": "", "valid_to": ""}


class EconomicEquityTests(unittest.TestCase):
    def test_acyclic_original_is_342_percent(self):
        result = economic_equity(load_from_equity_kg(), "C3")
        self.assertEqual(result["status"], "solved")
        self.assertAlmostEqual(result["owners"][0]["share"], .342)
        self.assertAlmostEqual(result["unknown_share"], .658)
        self.assertLess(result["residual"], 1e-10)

    def test_crossholding_solves_infinite_paths_not_simple_path_sum(self):
        case = EquityCase(entities=[Entity("P", "Person", "Person"), Entity("A", "A", "Company", capital=1000), Entity("B", "B", "Company", capital=1000)],
                          holdings=[Holding("P", "A", .8), Holding("B", "A", .2), Holding("A", "B", .5)])
        result = economic_equity(case, "B")
        self.assertAlmostEqual(result["owners"][0]["share"], .4 / .9)
        self.assertAlmostEqual(result["total_share"], 1)
        self.assertGreater(result["owners"][0]["share"], .4)

    def test_closed_100_percent_cycle_is_indeterminate(self):
        case = EquityCase(entities=[Entity("A", "A", "Company", capital=1000), Entity("B", "B", "Company", capital=1000)],
                          holdings=[Holding("A", "B", 1), Holding("B", "A", 1)])
        result = economic_equity(case, "A")
        self.assertEqual(result["status"], "indeterminate")
        self.assertIsNone(result["unknown_share"])
        self.assertFalse(result["owners"])

    def test_unrelated_singular_component_does_not_break_target(self):
        case = EquityCase(entities=[Entity("P", "P", "Person"), Entity("C", "C", "Company", capital=1000), Entity("A", "A", "Company", capital=1000), Entity("B", "B", "Company", capital=1000)],
                          holdings=[Holding("P", "C", 1), Holding("A", "B", 1), Holding("B", "A", 1)])
        self.assertEqual(economic_equity(case, "C")["owners"][0]["share"], 1)

    def test_invalid_person_target_and_missing_owner(self):
        with self.assertRaises(ValueError): economic_equity(load_from_equity_kg(), "P1")
        result = economic_equity(EquityCase(entities=[Entity("A", "A", "Company", capital=1000)]), "A")
        self.assertEqual(result["unknown_share"], 1)
        self.assertEqual(result["owners"], [])


class GovernanceTests(unittest.TestCase):
    def test_legacy_case_unchanged_with_empty_governance(self):
        data = payload()
        before = copy.deepcopy(data)
        result = governance_snapshot(data)
        self.assertEqual(data, before)
        self.assertIn(("Controls", "P1", "C3"), result["claims"])
        self.assertEqual(result["effective_case"], data["case"])

    def test_expiry_excludes_relation_and_withdraws_control(self):
        data = payload()
        data["relations"] = [{"key": "holdings:C2:C3", "evidence_id": "", "valid_from": "2026-10-01", "valid_to": "2026-10-23"}]
        before = governance_snapshot(data)
        after = governance_snapshot({**data, "as_of": "2026-10-23"})
        self.assertIn(("Controls", "P1", "C3"), before["claims"])
        self.assertNotIn(("Controls", "P1", "C3"), after["claims"])
        self.assertIn(("Controls", "P1", "C3"), claim_changes(before, after)["withdrawn"])
        self.assertEqual(after["excluded_relations"][0]["reason"], "关系不在有效期")

    def test_evidence_revocation_and_expiry_remove_dependency(self):
        data = payload()
        data["documents"] = [document(end="2026-10-23")]
        data["relations"] = [{"key": "holdings:C2:C3", "evidence_id": "ev1", "valid_from": "", "valid_to": ""}]
        result = governance_snapshot(data)
        self.assertTrue(result["documents"][0]["usable"])
        self.assertTrue(any("ev1" in f["source"] for f in result["input_facts"]))
        data["documents"][0]["status"] = "revoked"
        after = governance_snapshot(data)
        self.assertNotIn(("Controls", "P1", "C3"), after["claims"])

    def test_invalid_digest_and_unsupported_confirmation_rejected(self):
        data = payload(); data["documents"] = [document()]
        data["documents"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "摘要"): governance_snapshot(data)
        data["documents"][0]["sha256"] = ""; data["documents"][0]["reviewer"] = ""
        with self.assertRaisesRegex(ValueError, "核验人"): governance_snapshot(data)

    def test_unverified_document_never_supports_agreement(self):
        data = payload(); data["documents"] = [document()]; data["rights"] = [right(kind="agreement")]
        data["documents"][0]["status"] = "pending"
        result = governance_snapshot(data)
        self.assertNotIn(("Controls", "P3", "C3"), result["claims"])
        self.assertEqual(len(result["blocked_rights"]), 1)

    def test_explicit_votes_override_equity_assumption(self):
        data = payload(); data["documents"] = [document()]; data["rights"] = [right()]
        result = governance_snapshot(data)
        self.assertIn(("Controls", "P3", "C3"), result["claims"])
        self.assertNotIn(("Controls", "P1", "C3"), result["claims"])
        self.assertTrue(any(d["rule"] == "G01" for d in result["derivations"]))
        self.assertAlmostEqual(result["equity"]["owners"][0]["share"], .342)

    def test_agreement_control_and_expiry_restore_equity_rules(self):
        data = payload(); data["documents"] = [document()]; data["rights"] = [right(kind="agreement")]
        data["rights"][0]["valid_to"] = "2026-10-23"
        result = governance_snapshot(data)
        self.assertIn(("Controls", "P3", "C3"), result["claims"])
        expired = governance_snapshot({**data, "as_of": "2026-10-23"})
        self.assertIn(("Controls", "P1", "C3"), expired["claims"])
        self.assertNotIn(("Controls", "P3", "C3"), expired["claims"])

    def test_conflicting_agreements_do_not_choose_arbitrary_winner(self):
        data = payload(); data["documents"] = [document()]
        data["rights"] = [right(kind="agreement"), right("P2", "agreement", id_="r2")]
        result = governance_snapshot(data)
        self.assertEqual(len(result["conflicts"]), 1)
        self.assertEqual(result["accepted_rights"], [])
        self.assertFalse(any(f[0] == "Controls" and f[-1] == "C3" for f in result["claims"]))

    def test_overallocated_votes_are_conflict_not_normalized(self):
        data = payload(); data["documents"] = [document()]
        data["rights"] = [right(), right("P2", share=.6, id_="r2")]
        result = governance_snapshot(data)
        self.assertIn("100%", result["conflicts"][0]["reason"])
        self.assertFalse(result["accepted_rights"])

    def test_50_percent_votes_not_absolute_control(self):
        data = payload(); data["documents"] = [document()]; data["rights"] = [right(share=.5)]
        result = governance_snapshot(data)
        self.assertNotIn(("Controls", "P3", "C3"), result["claims"])
        self.assertFalse(result["conflicts"])

    def test_strict_modes_do_not_treat_missing_as_verified(self):
        data = payload(); data["strict_evidence"] = True
        result = governance_snapshot(data)
        self.assertEqual(len(result["effective_case"]["holdings"]), 0)
        self.assertEqual(result["equity"]["unknown_share"], 1)
        data["strict_evidence"] = False; data["strict_identity"] = True
        self.assertEqual(len(governance_snapshot(data)["effective_case"]["holdings"]), 0)

    def test_duplicate_identity_identifier_rejected(self):
        data = payload(); data["documents"] = [document()]
        data["identities"] = [{"entity_id": p, "identifier": "same", "status": "verified", "reviewer": "reviewer", "evidence_id": "ev1"} for p in ("P1", "P2")]
        with self.assertRaisesRegex(ValueError, "同一核验标识"): governance_snapshot(data)

    def test_inactive_windows_still_validate_each_date(self):
        data = payload()
        data["relations"] = [{"key": "holdings:C2:C3", "evidence_id": "", "valid_from": "2027-01-01", "valid_to": "not-a-date"}]
        with self.assertRaises(ValueError): governance_snapshot(data)

    def test_strict_identity_requires_both_right_parties(self):
        data = payload(); data["documents"] = [document()]; data["rights"] = [right()]
        data["strict_identity"] = True
        data["identities"] = [{"entity_id": "P3", "identifier": "demo-person", "status": "verified", "reviewer": "reviewer", "evidence_id": "ev1"}]
        result = governance_snapshot(data)
        self.assertFalse(result["accepted_rights"])
        self.assertEqual(len(result["blocked_rights"]), 1)
        data["identities"].append({"entity_id": "C3", "identifier": "demo-company", "status": "verified", "reviewer": "reviewer", "evidence_id": "ev1"})
        self.assertIn(("Controls", "P3", "C3"), governance_snapshot(data)["claims"])

    def test_expired_trade_evidence_is_unknown_not_new_shell(self):
        data = payload()
        data["relations"] = [{"key": "trades:" + t["seller"] + ":" + t["buyer"], "evidence_id": "", "valid_from": "", "valid_to": "2026-10-22"} for t in data["case"]["trades"]]
        result = governance_snapshot(data)
        self.assertTrue(result["missing_trade_evidence"])
        self.assertFalse(any(f["fact"][0] == "Shell" for f in result["input_facts"]))

    def test_invalid_dates_duplicate_ids_and_unknown_relations(self):
        for modified in ({"as_of": "2026-02-30"}, {"documents": [document(), document()]},
            {"relations": [{"key": "holdings:NO:C3", "evidence_id": "", "valid_from": "", "valid_to": ""}]}):
            with self.assertRaises(ValueError): governance_snapshot({**payload(), **modified})
        data = payload(); data["documents"] = [document(end="2026-10-01")]
        data["documents"][0]["valid_from"] = "2026-10-22"
        with self.assertRaises(ValueError): governance_snapshot(data)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="equity_history_tests_")
        self.base = Path(self.temp.name).resolve()
        self.assertEqual(self.base.parent, Path(tempfile.gettempdir()).resolve())
        self.store = GovernanceHistory(self.base / "history.sqlite3")

    def tearDown(self):
        self.assertEqual(self.base.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.base.name.startswith("equity_history_tests_"))
        self.temp.cleanup()

    def test_persistent_versions_and_auto_expiry_evaluation(self):
        data = payload(); data["relations"] = [{"key": "holdings:C2:C3", "evidence_id": "", "valid_from": "", "valid_to": "2026-10-23"}]
        saved = self.store.save("  test  ", data)
        reopened = GovernanceHistory(self.base / "history.sqlite3")
        result = reopened.evaluate(saved["case_id"], "2026-10-23")
        self.assertIn(("Controls", "P1", "C3"), result["changes"]["withdrawn"])
        self.assertEqual(reopened.load(saved["case_id"])["name"], "test")
        self.assertEqual(reopened.load(saved["case_id"])["payload"]["as_of"], "2026-10-22")
        second = reopened.save("test", result["payload"], saved["case_id"], 1)
        self.assertEqual(second["revision"], 2)
        self.assertEqual(len(reopened.history(saved["case_id"])), 2)
        self.assertEqual(reopened.load(saved["case_id"], 1)["revision"], 1)

    def test_optimistic_conflict_keeps_history(self):
        saved = self.store.save("test", payload())
        with self.assertRaises(RevisionConflict): self.store.save("test", payload(), saved["case_id"], 0)
        self.assertEqual(len(self.store.history(saved["case_id"])), 1)

    def test_hash_chain_tamper_is_detected(self):
        saved = self.store.save("test", payload())
        with self.store._connect() as connection:
            connection.execute("UPDATE revisions SET payload='{}' WHERE case_id=?", (saved["case_id"],))
        with self.assertRaisesRegex(ValueError, "哈希链"): self.store.load(saved["case_id"])

    def test_invalid_ids_and_revision_never_access_arbitrary_path(self):
        with self.assertRaises(ValueError): self.store.load("../../secret")
        saved = self.store.save("test", payload())
        with self.assertRaises(ValueError): self.store.load(saved["case_id"], 0)
        self.assertEqual(len(self.store.list_cases()), 1)


try:
    from fastapi.testclient import TestClient
    from backend.api import create_app
    from backend.service import InferenceService
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False


@unittest.skipUnless(WEB_AVAILABLE, "optional web requirements")
class GovernanceApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="equity_history_api_")
        self.base = Path(self.temp.name).resolve()
        self.client = TestClient(create_app(governance_database=self.base / "api.sqlite3", governance_runner=governance_snapshot))

    def tearDown(self):
        self.client.close()
        self.assertEqual(self.base.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.base.name.startswith("equity_history_api_"))
        self.temp.cleanup()

    def test_preview_save_load_expire_history(self):
        data = payload(); data["relations"] = [{"key": "holdings:C2:C3", "evidence_id": "", "valid_from": "", "valid_to": "2026-10-23"}]
        preview = self.client.post("/api/governance/preview", json=data)
        self.assertEqual(preview.status_code, 200, preview.text)
        saved = self.client.post("/api/governance/save", json={"name": "test", "payload": data}).json()
        loaded = self.client.post("/api/governance/load", json={"case_id": saved["case_id"], "as_of": "2026-10-23"})
        self.assertEqual(loaded.status_code, 200, loaded.text)
        self.assertIn(["Controls", "P1", "C3"], loaded.json()["changes"]["withdrawn"])
        self.assertEqual(len(self.client.get("/api/governance/cases").json()["items"]), 1)
        conflict = self.client.post("/api/governance/save", json={"name": "test", "payload": data, "case_id": saved["case_id"], "expected_revision": 0})
        self.assertEqual(conflict.status_code, 409)

    def test_import_recomputes_only_validated_input_and_does_not_persist(self):
        data = payload()
        response = self.client.post("/api/governance/import", json=data)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertFalse(result["persisted"])
        self.assertIn(["Controls", "P1", "C3"], result["snapshot"]["claims"])
        self.assertEqual(result["payload"], data)
        self.assertEqual(self.client.get("/api/governance/cases").json()["items"], [])
        self.assertEqual(self.client.post("/api/governance/import", json={**data, "snapshot": {"claims": [["Controls", "P3", "C3"]]}}).status_code, 422)
        self.assertEqual(self.client.post("/api/governance/import", json={**data, "target": "NO"}).status_code, 422)

    def test_cycle_example_computes_444444_percent(self):
        response = self.client.get("/api/governance/example")
        self.assertEqual(response.status_code, 200, response.text)
        example = response.json()
        data = {**payload(), "case": example["case"], "target": example["target"]}
        result = self.client.post("/api/governance/preview", json=data).json()
        self.assertAlmostEqual(result["equity"]["owners"][0]["share"], example["expected_person_share"])

    def test_labels_types_unknown_fields_and_tampered_evidence_rejected(self):
        for change in ({"strict_evidence": "true"}, {"as_of": "yesterday"}, {"target": "NO"}, {"file_path": "../../auth.json"}):
            result = self.client.post("/api/governance/preview", json={**payload(), **change})
            self.assertEqual(result.status_code, 422, result.text)
        data = payload(); data["documents"] = [document()]; data["documents"][0]["sha256"] = "0" * 64
        self.assertEqual(self.client.post("/api/governance/preview", json=data).status_code, 422)
