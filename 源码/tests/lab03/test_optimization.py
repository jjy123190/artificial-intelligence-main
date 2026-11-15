"""Regression tests for depth-aware features, cost accounting and LLM prompt selection."""
from tests.lab03.support import requires_artifacts
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from experiments.lab03.domain import Graph, Node, Edge, Query, State, features
from experiments.lab03.search import search, FIXED, HYBRID_ML_WEIGHT
from experiments.lab03.llm import prompt_for, legality_prompt_for, LEGALITY_TOKENS, EVIDENCE_POLICY
from experiments.lab03.dataset import load_dataset, unpack, ROOT


def fixture():
    graph = Graph([Node('p', 'Start', kind='Person'), Node('a', 'Other sector'),
                   Node('bad', 'Decoy', sector='制造', listed=True),
                   Node('z', 'Endpoint', sector='制造', listed=True)],
                  [Edge('first', 'p', 'a', 'HOLDS'), Edge('bad', 'p', 'bad', 'HOLDS'),
                   Edge('last', 'a', 'z', 'HOLDS')])
    return graph, Query('q', 'p', ('HOLDS', 'HOLDS'), '制造')


class OneHopOptimizationTests(TestCase):
    def test_observable_neighbor_prevents_premature_sector_bias(self):
        graph, query = fixture()
        result = search(graph, query, 'astar_fixed', 3)
        self.assertTrue(result['valid'])
        self.assertEqual(result['edge_ids'], ['first', 'last'])

    def test_only_one_next_relation_neighborhood_is_read(self):
        graph, query = fixture()
        query = Query('q', 'p', ('HOLDS',) * 4, '制造')
        with patch.object(graph, 'successors', wraps=graph.successors) as successors, \
             patch('experiments.lab03.domain.enumerate_paths', side_effect=AssertionError('oracle leaked')):
            features(graph, State(('p',)), query)
        self.assertEqual(successors.call_count, 1)
        self.assertEqual(successors.call_args.args[0].depth, 0)

    def test_expert_toggle_is_respected_by_features_not_just_expansion(self):
        graph, query = fixture()
        graph = Graph(list(graph.nodes.values()), [Edge('first', 'p', 'a', 'HOLDS', verified=False), graph.edges['last']])
        self.assertEqual(features(graph, State(('p',)), query, True)[5], 1)
        self.assertEqual(features(graph, State(('p',)), query, False)[5], 0)
        self.assertEqual(features(graph, State(('p',)), query, True)[2], 0)
        self.assertGreater(features(graph, State(('p',)), query, False)[2], 0)

    def test_onehop_endpoint_checks_every_public_predicate(self):
        graph, query = fixture()
        state = graph.state_for_edges(query, ['first'])
        for replacement in [Node('z', 'Endpoint', kind='Person', sector='制造', listed=True),
                            Node('z', 'Endpoint', sector='其他', listed=True),
                            Node('z', 'Endpoint', sector='制造', listed=False)]:
            changed = Graph([replacement if n.id == 'z' else n for n in graph.nodes.values()], graph.edges.values())
            self.assertEqual(features(changed, state, query)[1], 0)
        self.assertEqual(features(graph, state, query)[1], 1)

    def test_dead_end_uses_next_relation_and_avoids_path_cycles(self):
        graph, query = fixture()
        state = graph.state_for_edges(query, ['first'])
        graph = Graph(list(graph.nodes.values()), [graph.edges['first'], Edge('cycle', 'a', 'p', 'HOLDS'),
                                                  Edge('wrong', 'a', 'z', 'TRADES')])
        x = features(graph, state, query)
        self.assertEqual(x[2], 0)
        self.assertEqual(x[5], 1)

    def test_admissible_and_blind_kernels_never_compute_features(self):
        graph, query = fixture()
        for method in ['bfs', 'mcts', 'astar_admissible']:
            with self.subTest(method=method), patch('experiments.lab03.search.features', side_effect=AssertionError('unnecessary lookahead')):
                result = search(graph, query, method, 16)
            self.assertEqual(result['heuristic_evaluations'], 0)
            self.assertEqual(result['heuristic_edge_checks'], 0)
            self.assertEqual(result['total_edge_checks'], result['successor_checks'])

    def test_feature_and_heuristic_cache_are_per_full_prefix(self):
        graph, query = fixture()
        calls = []
        def checked(*args, **kwargs):
            calls.append(args[1].edges)
            return features(*args, **kwargs)
        with patch('experiments.lab03.search.features', side_effect=checked):
            result = search(graph, query, 'astar_fixed', 16, trace=True)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(result['heuristic_evaluations'], len(calls))
        self.assertGreater(result['heuristic_edge_checks'], 0)
        self.assertEqual(result['total_edge_checks'], result['successor_checks'] + result['heuristic_edge_checks'])

    def test_tracing_does_not_add_hidden_search_work(self):
        graph, query = fixture()
        model = {'coef': [0.] * 6, 'intercept': 0., 'seed': 7}
        for method in ['bfs', 'astar_admissible', 'astar_fixed', 'astar_ga', 'astar_ml', 'hybrid', 'hybrid_no_expert', 'mcts']:
            traced = search(graph, query, method, 16, FIXED, model, trace=True)
            plain = search(graph, query, method, 16, FIXED, model, trace=False)
            for key in ['expanded', 'edge_ids', 'successor_checks', 'heuristic_edge_checks', 'heuristic_evaluations', 'total_edge_checks']:
                with self.subTest(method=method, key=key):
                    self.assertEqual(traced[key], plain[key])

    @requires_artifacts('optimization/protocol.json')
    def test_validation_protocol_is_disjoint_and_keeps_learning_active(self):
        folder = ROOT / 'outputs/lab03/optimization'
        protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
        self.assertEqual(protocol['selection_split'], 'validation')
        self.assertEqual(protocol['classifier_fit_split'], 'train')
        self.assertEqual(protocol['selected']['hybrid_ml_weight'], HYBRID_ML_WEIGHT)
        self.assertGreater(HYBRID_ML_WEIGHT, 0)
        rows = json.loads((folder/'validation-llm-sweep.json').read_text(encoding='utf-8'))
        self.assertTrue(all(r['query'].startswith('validation-') for v in rows for r in v['records']))


class BooleanLegalityPromptTests(TestCase):
    def test_prompt_copies_raw_prefix_facts_and_not_expected_label(self):
        graph, query = fixture()
        prompt = prompt_for(graph, query, graph.edges['first'])
        text = legality_prompt_for(prompt)
        self.assertIn(EVIDENCE_POLICY, text)
        self.assertIn('verified=True; active=True; minimum_confidence=1.0', text)
        self.assertNotIn('expert_legal', text)
        self.assertNotIn('Expected', text)
        self.assertEqual(LEGALITY_TOKENS, ('True', 'False'))

    def test_earlier_bad_edge_is_still_visible_in_legality_prompt(self):
        graph, query = fixture()
        graph = Graph(graph.nodes.values(), [Edge('first', 'p', 'a', 'HOLDS', verified=False), graph.edges['last']])
        state = graph.state_for_edges(query, ['first'])
        prompt = prompt_for(graph, query, graph.edges['last'], state)
        self.assertIn('verified=False; active=True;', legality_prompt_for(prompt))

    def test_malformed_or_nonfinite_prefix_facts_rejected(self):
        for text in ['Candidate: fake', 'Prefix verified=True; active=True; minimum confidence=9.0.\n',
                     'Prefix verified=True; active=True; minimum confidence=1e999.\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                legality_prompt_for(text)


class OptimizationEvidenceTests(TestCase):
    @requires_artifacts('models.json','ga_history.json','manifest.json','optimization/comparison.json','optimization/baseline_sources.zip')
    def test_saved_comparison_recomputes_and_retains_regressions(self):
        from experiments.lab03.optimization import audit_optimization
        result = audit_optimization()
        self.assertEqual(result['same_question_pairs'], 3888)
        self.assertEqual(len(result['comparisons']), 48)
        self.assertTrue(result['regressions'])
        self.assertEqual(result, json.loads((ROOT/'outputs/lab03/optimization/comparison.json').read_text(encoding='utf-8')))

    def test_duplicate_runs_cannot_be_hidden_by_dictionary_indexing(self):
        import tempfile
        from experiments.lab03.optimization import csv_rows
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/'duplicate.csv'
            file.write_text('query_id,budget,method\nq,8,bfs\nq,8,bfs\n', encoding='utf-8')
            with self.assertRaises(AssertionError):
                csv_rows(file)
