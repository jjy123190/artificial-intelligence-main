"""Regression tests for failures not covered by the original course happy paths."""
import copy
import unittest
from unittest.mock import patch

from experiments.lab03.domain import Graph, Node, Edge, Query, State
from experiments.lab03.search import search
from experiments.lab03 import service
from tests.lab03.test_search import fixture


class SearchBoundaryTests(unittest.TestCase):
    def test_noninteger_budget_cannot_exceed_resource_bound(self):
        graph, query = fixture()
        for budget in [True, False, 1.5, "8", float("nan"), float("inf")]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                search(graph, query, budget=budget)

    def test_unknown_relation_is_not_a_logical_no_answer(self):
        graph, query = fixture()
        with self.assertRaisesRegex(ValueError, "未知关系"):
            search(graph, Query("x", query.start, ("invented", "HOLDS"), "制造"))

    def test_rejects_nonfinite_or_incomplete_model_parameters(self):
        graph, query = fixture()
        for weights in [[], [0] * 5, [True] * 6, [float("nan")] * 6, [-1] * 6]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                search(graph, query, weights=weights)
        for model in [{}, {"coef": [0] * 5, "intercept": 0}, {"coef": [0] * 6, "intercept": float("inf")}]:
            with self.subTest(model=model), self.assertRaises(ValueError):
                search(graph, query, model=model)

    def test_zero_weights_are_preserved(self):
        graph, query = fixture()
        result = search(graph, query, "astar_ga", 16, weights=[0] * 6)
        self.assertEqual(result["trace"][0]["h"], query.hops)
        self.assertTrue(result["valid"])

    def test_trace_does_not_claim_bfs_or_mcts_uses_a_star_heuristic(self):
        graph, query = fixture()
        for method in ["bfs", "mcts"]:
            result = search(graph, query, method, 16)
            self.assertFalse(result["heuristic_used"])
            self.assertTrue(all(s["h"] is None and s["f"] is None and s["p"] is None for s in result["trace"]))

    def test_parent_path_remains_correct_on_expansion_and_solution(self):
        graph, query = fixture()
        result = search(graph, query, "astar_fixed", 16)
        for step in result["trace"]:
            self.assertEqual(step["parent_path"], step["edge_ids"][:-1])
            self.assertAlmostEqual(step["f"], step["g"] + step["h"], places=4)
        self.assertEqual(result["trace_events"], len(result["trace"]))
        self.assertFalse(result["trace_truncated"])

    def test_truncated_trace_keeps_solution_and_discloses_omitted_events(self):
        nodes = [Node("p", "起点", kind="Person"), Node("a", "中间"), Node("z", "终点", sector="制造", listed=True)]
        edges = [Edge(f"e{i:04}", "p", "a", "HOLDS", verified=False) for i in range(1700)]
        edges += [Edge("good", "p", "a", "HOLDS"), Edge("last", "a", "z", "HOLDS")]
        result = search(Graph(nodes, edges), Query("q", "p", ("HOLDS", "HOLDS"), "制造"), "bfs", 8)
        self.assertTrue(result["valid"])
        self.assertTrue(result["trace_truncated"])
        self.assertGreater(result["trace_events"], 1600)
        self.assertEqual(len(result["trace"]), 1600)
        self.assertEqual(result["trace"][-1]["status"], "solution")
        self.assertEqual(result["trace"][-1]["edge_ids"], result["edge_ids"])

    def test_malformed_witness_returns_false_instead_of_crashing(self):
        graph, query = fixture()
        cases = [State(()), State(("p", "a", "b", "missing"), ("e1", "e2", "e3")),
                 State(("p", "a", "b", "c", "x"), ("e1", "e2", "e3", "bad")),
                 State(("p", "missing", "b", "c"), ("e1", "e2", "e3"))]
        for state in cases:
            with self.subTest(state=state):
                self.assertFalse(graph.valid_path(state, query))

    def test_llm_scores_cannot_smuggle_unknown_edges_or_nan(self):
        graph, query = fixture()
        for scores in [{"made-up": 1.0}, {"e1": float("nan")}, {"e1": True}, {"e1": -0.1}]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                search(graph, query, "hybrid_llm", llm_scores=scores)

    def test_budget_boundary_and_exhaustion_are_consistent(self):
        graph = Graph([Node("p", "人", kind="Person"), Node("a", "甲"), Node("c", "目标", sector="制造", listed=True)],
                      [Edge("e1", "p", "a", "HOLDS"), Edge("e2", "a", "c", "HOLDS")])
        query = Query("q", "p", ("HOLDS", "HOLDS"), "制造")
        for method in ["bfs", "astar_admissible", "astar_fixed"]:
            before = search(graph, query, method, 2)
            exact = search(graph, query, method, 3)
            self.assertEqual(before["status"], "budget_exhausted")
            self.assertTrue(exact["valid"])
            self.assertEqual(exact["expanded"], 3)
        empty = Graph(list(graph.nodes.values()), [])
        for method in ["bfs", "astar_fixed", "mcts"]:
            self.assertEqual(search(empty, query, method, 1)["status"], "exhausted")

    def test_small_random_graphs_match_an_independent_exhaustive_oracle(self):
        import random
        from experiments.lab03.domain import enumerate_paths
        for seed in range(60):
            rng = random.Random(seed)
            nodes = [Node(str(i), str(i), sector="target" if rng.random() < .5 else "other", listed=rng.random() < .5) for i in range(7)]
            edges = []
            for source in nodes:
                for target in nodes:
                    if source != target and rng.random() < .18:
                        edges.append(Edge(str(len(edges)), source.id, target.id, rng.choice(["HOLDS", "TRADES"]),
                                          confidence=rng.choice([.4, .8, 1.]), verified=rng.random() > .1, active=rng.random() > .1))
            graph = Graph(nodes, edges)
            query = Query(str(seed), "0", tuple(rng.choice(["HOLDS", "TRADES"]) for _ in range(rng.choice([2, 3, 4]))), "target")
            expected = bool(enumerate_paths(graph, query))
            for method in ["bfs", "astar_admissible", "astar_fixed", "astar_ga", "astar_ml", "hybrid", "mcts"]:
                with self.subTest(seed=seed, method=method):
                    result = search(graph, query, method, 256, trace=False)
                    self.assertEqual(result["valid"], expected)
                    self.assertNotEqual(result["status"], "budget_exhausted")


class ServiceIntegrityTests(unittest.TestCase):
    def modify_artifact(self, name, transform):
        original = service.read
        def read(filename, default=None):
            value = original(filename, default)
            return transform(copy.deepcopy(value)) if filename == name else value
        return patch.object(service, "read", side_effect=read)

    def test_baselines_do_not_depend_on_learned_artifacts(self):
        with patch.object(service, "read", side_effect=AssertionError("baseline read a model")):
            for method in ["bfs", "astar_fixed", "astar_admissible", "mcts"]:
                self.assertTrue(service.run_task({"method": method, "budget": 128})["result"]["valid"])

    def test_missing_classifier_is_not_silently_neutral(self):
        with self.modify_artifact("models.json", lambda _: {}):
            for method in ["astar_ml", "hybrid", "hybrid_no_expert", "hybrid_llm"]:
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, "分类器"):
                    service.run_task({"method": method})

    def test_missing_ga_is_not_silently_fixed(self):
        with self.modify_artifact("ga_history.json", lambda _: {}):
            for method in ["astar_ga", "hybrid"]:
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, "GA参数"):
                    service.run_task({"method": method})

    def test_corrupt_optional_files_do_not_break_baselines_or_snapshot(self):
        original = service.read
        def read(name, default=None):
            if name in {"models.json", "ga_history.json"}: raise ValueError("损坏")
            return original(name, default)
        with patch.object(service, "read", side_effect=read):
            snapshot = service.snapshot()
            self.assertGreater(len(snapshot["warnings"]), 0)
            self.assertTrue(snapshot["queries"])
            self.assertIn("hybrid", snapshot["queries"][0]["unavailable_methods"])
            self.assertTrue(service.run_task({"method": "bfs", "budget": 128})["result"]["valid"])

    def test_model_and_ga_metadata_must_match_the_query_seed(self):
        for name in ["models.json", "ga_history.json"]:
            def corrupt(value):
                value["7"]["seed"] = 19
                return value
            with self.modify_artifact(name, corrupt), self.assertRaises(ValueError):
                service.run_task({"method": "hybrid"})

    def test_malformed_model_metadata_is_rejected_not_crashed(self):
        for name, key in [("models.json", "train_graph_ids"), ("ga_history.json", "validation_graph_ids")]:
            def corrupt(value):
                value["7"][key] = 7
                return value
            with self.modify_artifact(name, corrupt):
                with self.assertRaises(ValueError):
                    service.run_task({"method": "hybrid"})
                self.assertTrue(service.snapshot()["warnings"])

    def test_replay_scores_require_valid_structure_not_just_hashes(self):
        for scores in [{"missing": 0.4}, {"edge": float("nan")}, {}, []]:
            def corrupt(value):
                value["test-7-0-Q0"]["scores"] = scores
                return value
            with self.subTest(scores=scores), self.modify_artifact("llm_cache.json", corrupt):
                with self.assertRaises(ValueError):
                    service.run_task({"query_id": "test-7-0-Q0", "method": "hybrid_llm"})
                query = next(q for q in service.snapshot()["queries"] if q["id"] == "test-7-0-Q0")
                self.assertFalse(query["llm_cached"])

    def test_missing_replay_fields_are_explained_not_keyerrors(self):
        def corrupt(value):
            value["test-7-0-Q0"].pop("graph_sha256")
            return value
        with self.modify_artifact("llm_cache.json", corrupt), self.assertRaisesRegex(ValueError, "不一致"):
            service.run_task({"query_id": "test-7-0-Q0", "method": "hybrid_llm"})

    def test_failure_explanations_distinguish_budget_exhaustion_and_invalid_evidence(self):
        graph, query = fixture()
        for method, budget, expected in [("bfs", 1, "预算"), ("bfs", 128, "全部检查"), ("hybrid_no_expert", 128, "证据未通过")]:
            q = Query("q", "p", ("HOLDS", "HOLDS"), "制造")
            result = search(graph, q, method, budget)
            self.assertIn(expected, service.explain_result(graph, result))

    def test_low_confidence_edge_is_never_presented_as_eligible(self):
        graph = Graph([Node("p", "人"), Node("a", "甲"), Node("c", "目标", sector="制造", listed=True)],
                      [Edge("e1", "p", "a", "HOLDS", confidence=.54), Edge("e2", "a", "c", "HOLDS")])
        query = Query("q", "p", ("HOLDS", "HOLDS"), "制造")
        with patch.object(service, "tasks", return_value={"q": (graph, query, 7)}):
            result = service.run_task({"query_id": "q", "method": "hybrid_no_expert", "budget": 16})
        self.assertTrue(result["result"]["found"])
        self.assertFalse(result["result"]["valid"])
        self.assertFalse(result["path_evidence"][0]["eligible"])
        self.assertIn("置信度", result["path_evidence"][0]["rejection_reason"])

    def test_snapshot_never_returns_remote_credentials(self):
        import os
        with patch.dict(os.environ, {"LAB03_LLM_API_KEY": "secret-sentinel", "LAB03_LLM_MODEL": "model-sentinel"}):
            value = service.snapshot()
            self.assertTrue(value["remote_available"])
            self.assertNotIn("secret-sentinel", str(value))


class LabThreeAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from backend.api import create_app
        cls.client = TestClient(create_app())

    def test_snapshot_has_honest_availability_and_no_secrets(self):
        response = self.client.get("/api/labs/lab03/snapshot")
        self.assertEqual(response.status_code, 200)
        value = response.json()
        self.assertIsInstance(value["remote_available"], bool)
        self.assertEqual(len(value["queries"]), 12)
        self.assertTrue(all("unavailable_methods" in q for q in value["queries"]))
        self.assertIn("no-store", response.headers["cache-control"])

    def test_unreadable_snapshot_is_actionable_json_not_server_crash(self):
        with patch("experiments.lab03.service.snapshot", side_effect=ValueError("damaged file")):
            response = self.client.get("/api/labs/lab03/snapshot")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "lab03_data_unavailable")

    def test_remote_mode_flag_cannot_extend_non_llm_resource_budget(self):
        with patch("backend.api.run_bounded", return_value={"ok": True}) as runner:
            response = self.client.post("/api/labs/lab03/run", json={"method": "bfs", "llm_mode": "remote"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(runner.call_args.args[-1], 15.0)
            self.client.post("/api/labs/lab03/run", json={"method": "hybrid_llm", "llm_mode": "remote"})
            self.assertEqual(runner.call_args.args[-1], 180.0)

    def test_api_rejects_forged_inputs_before_starting_worker(self):
        with patch("backend.api.run_bounded") as runner:
            for payload in [{"budget": True}, {"budget": 1.5}, {"budget": 129}, {"query_id": ""}, {"method": "unknown"}, {"graph": {}}, {"answer": "c"}]:
                with self.subTest(payload=payload):
                    response = self.client.post("/api/labs/lab03/run", json=payload)
                    self.assertEqual(response.status_code, 422)
            runner.assert_not_called()

    def test_error_releases_worker_slot_for_retry(self):
        from backend.service import WebError
        with patch("backend.api.run_bounded", side_effect=WebError("test_error", "可重试", 422)):
            for _ in range(3):
                self.assertEqual(self.client.post("/api/labs/lab03/run", json={}).status_code, 422)
        with patch("backend.api.run_bounded", return_value={"ok": True}):
            self.assertEqual(self.client.post("/api/labs/lab03/run", json={}).status_code, 200)

    def test_course_does_not_lose_existing_lab_entries(self):
        value = self.client.get("/api/course").json()
        self.assertEqual([lab["id"] for lab in value["labs"]], ["lab01", "lab02", "lab03"])


if __name__ == "__main__":
    unittest.main()
