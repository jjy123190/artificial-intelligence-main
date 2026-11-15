"""Regressions for learned-artifact contracts and independent kernel completeness."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import tempfile
from unittest import TestCase
from unittest.mock import patch

from experiments.lab03 import service, learned_integrity
from experiments.lab03.dataset import load_dataset
from experiments.lab03.domain import Edge, Graph, Node, Query, path_key
from experiments.lab03.learning import classifier_evaluation
from experiments.lab03.llm import candidate_actions
from experiments.lab03.runtime_integrity import clear_verified_inputs
from experiments.lab03.search import FIXED, METHODS, search
from tests.lab03.support import artifact_fixture, requires_artifacts


class LearnedContractTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = artifact_fixture('models.json')
        cls.ga = artifact_fixture('ga_history.json')
        cls.records = load_dataset()

    def setUp(self):
        clear_verified_inputs()

    def test_real_training_and_validation_contracts_are_accepted(self):
        for seed in (7, 19, 43):
            self.assertTrue(service.valid_model(self.models[str(seed)], seed))
            self.assertTrue(service.valid_ga(self.ga[str(seed)], seed))

    def test_ga_requires_complete_contiguous_unique_generations(self):
        for change in (lambda g: g.update(generations=1),
                       lambda g: g['history'][0].update(generation=True),
                       lambda g: g['history'][0].update(generation=1),
                       lambda g: g.update(history=[])):
            value = copy.deepcopy(self.ga['7']); change(value)
            self.assertFalse(service.valid_ga(value, 7))

    def test_ga_deployed_weights_must_equal_last_champion(self):
        value = copy.deepcopy(self.ga['7'])
        value['weights'][0] = (value['weights'][0] + 1) % 5
        self.assertFalse(service.valid_ga(value, 7))

    def test_ga_needs_bounded_weights_with_disabled_classifier_gene(self):
        for weights in ([1.] * 6, [1.] * 5 + [True], [6.] * 5 + [0.], [0.] * 5,
                        [float('nan')] + [0.] * 5):
            value = copy.deepcopy(self.ga['7']); value['weights'] = weights
            self.assertFalse(service.valid_ga(value, 7))

    def test_ga_fitness_must_match_its_observations(self):
        value = copy.deepcopy(self.ga['7']); value['history'][0]['fitness'] = 42.
        self.assertFalse(service.valid_ga(value, 7))

    def test_ga_consistent_but_invented_metrics_fail_validation_replay(self):
        value = copy.deepcopy(self.ga['7'])
        row = value['history'][0]
        row['success_rate'] = 0. if row['success_rate'] else 1.
        row['fitness'] = row['success_rate'] - .12 * row['expanded_fraction']
        learned_integrity.ga_schema(value, 7, self.records)
        self.assertFalse(service.valid_ga(value, 7))

    def test_ga_replay_uses_validation_only_and_does_not_reoptimize(self):
        with (patch('experiments.lab03.genetic.optimize', side_effect=AssertionError('must not tune')),
              patch.object(learned_integrity, 'search', wraps=search) as calls):
            learned_integrity.ga_validation_evidence(self.ga['7'], 7, self.records)
        self.assertTrue(calls.call_count)
        for call in calls.call_args_list:
            graph, query, method, budget = call.args
            self.assertEqual(graph.split, 'validation')
            self.assertEqual(method, 'astar_ga')

    def test_ga_rejects_wrong_validation_scope_or_question_count(self):
        for field, value in [('validation_graph_ids', ['validation-7-fabricated']),
                             ('validation_graph_ids', self.ga['7']['validation_graph_ids'] * 2),
                             ('validation_queries', 1), ('validation_queries', True),
                             ('population', 2), ('budget', True), ('unique_candidates', 999999),
                             ('fitness', 'test accuracy')]:
            changed = copy.deepcopy(self.ga['7']); changed[field] = value
            with self.subTest(field=field, value=value):
                self.assertFalse(service.valid_ga(changed, 7))

    def test_classifier_rejects_nontraining_graphs_missing_failures_or_wrong_features(self):
        for field, value in [('kind', 'unit_test_fixture'), ('train_graph_ids', ['train-7-fabricated']),
                             ('negative', 0), ('positive', True), ('samples', 1),
                             ('features', ['identifier'] * 6), ('train_brier', '0.1')]:
            changed = copy.deepcopy(self.models['7']); changed[field] = value
            with self.subTest(field=field):
                self.assertFalse(service.valid_model(changed, 7))

    def test_classifier_finite_parameter_edit_cannot_retain_saved_training_error(self):
        value = copy.deepcopy(self.models['7']); value['intercept'] += 3.
        self.assertFalse(service.valid_model(value, 7))

    def test_classifier_plausible_but_wrong_sample_counts_fail_training_reconstruction(self):
        value = copy.deepcopy(self.models['7']); value['samples'] += 1; value['negative'] += 1
        self.assertFalse(service.valid_model(value, 7))

    def test_classifier_validation_reads_only_training_prefixes(self):
        original = learned_integrity.enumerate_paths
        def checked(graph, query):
            self.assertEqual(graph.split, 'train')
            return original(graph, query)
        with patch.object(learned_integrity, 'enumerate_paths', side_effect=checked), \
             patch('experiments.lab03.learning.fit_classifier', side_effect=AssertionError('must not refit')):
            self.assertTrue(service.valid_model(self.models['7'], 7))

    def test_successful_checks_cache_full_content_not_mutable_response(self):
        value = copy.deepcopy(self.ga['7'])
        with patch.object(learned_integrity, 'ga_validation_evidence',
                          wraps=learned_integrity.ga_validation_evidence) as check:
            self.assertTrue(service.valid_ga(value, 7))
            self.assertTrue(service.valid_ga(copy.deepcopy(value), 7))
            self.assertEqual(check.call_count, 1)
            row = value['history'][0]
            row['success_rate'] = 0. if row['success_rate'] else 1.
            row['fitness'] = row['success_rate'] - .12 * row['expanded_fraction']
            self.assertFalse(service.valid_ga(value, 7))
            self.assertEqual(check.call_count, 2)

    def test_valid_metrics_recompute_against_test_only_without_refitting(self):
        rows = [{'seed': seed, **classifier_evaluation(self.records, self.models[str(seed)])}
                for seed in (7, 19, 43)]
        with patch('experiments.lab03.learning.fit_classifier', side_effect=AssertionError('must not fit')):
            learned_integrity.learning_metrics(rows, self.models, self.records)

    def test_metrics_reject_malformed_duplicate_or_plausibly_changed_values(self):
        original = [{'seed': seed, **classifier_evaluation(self.records, self.models[str(seed)])}
                    for seed in (7, 19, 43)]
        for change in (lambda rows: rows[0].update(f1='1.0'),
                       lambda rows: rows[0].update(f1=True),
                       lambda rows: rows[0].update(f1=.5),
                       lambda rows: rows[0].update(split='train'),
                       lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
                       lambda rows: rows.pop()):
            rows = copy.deepcopy(original); change(rows)
            with self.assertRaises(ValueError):
                learned_integrity.learning_metrics(rows, self.models, self.records)


@requires_artifacts('models.json', 'ga_history.json', 'manifest.json', 'learning_metrics.json')
class LearnedDisplayTests(TestCase):
    def setUp(self):
        clear_verified_inputs()

    def corrupt(self, name, transform):
        original = service.read
        def load(filename, default=None):
            value = original(filename, default)
            return transform(copy.deepcopy(value)) if filename == name else value
        return patch.object(service, 'read', side_effect=load)

    def test_string_metric_is_hidden_instead_of_crashing_frontend_to_fixed(self):
        def change(rows): rows[0]['f1'] = 'corrupt'; return rows
        with self.corrupt('learning_metrics.json', change):
            snapshot = service.snapshot()
        self.assertEqual(snapshot['learning'], [])
        self.assertTrue(any('分类器测试' in w for w in snapshot['warnings']))
        self.assertTrue(snapshot['summary'])
        json.dumps(snapshot, allow_nan=False)

    def test_empty_metrics_are_reported_as_missing_not_silently_accepted(self):
        with self.corrupt('learning_metrics.json', lambda _: []):
            snapshot = service.snapshot()
        self.assertEqual(snapshot['learning'], [])
        self.assertTrue(any('分类器测试' in w for w in snapshot['warnings']))
        self.assertTrue(snapshot['summary'])

    def test_finite_invented_metric_is_not_presented_as_measured(self):
        def change(rows): rows[0]['f1'] = .5; return rows
        with self.corrupt('learning_metrics.json', change):
            self.assertEqual(service.snapshot()['learning'], [])

    def test_corrupt_ga_cannot_run_or_publish_curve_and_stale_search_table(self):
        def change(rows): rows['7']['history'] = rows['7']['history'][:1]; return rows
        with self.corrupt('ga_history.json', change):
            with self.assertRaisesRegex(ValueError, 'GA参数'):
                service.run_task({'method': 'astar_ga'})
            snapshot = service.snapshot()
            self.assertTrue(service.run_task({'method': 'bfs', 'budget': 128})['result']['valid'])
        self.assertNotIn('7', snapshot['ga'])
        self.assertEqual(snapshot['summary'], [])
        self.assertIn('astar_ga', snapshot['queries'][0]['unavailable_methods'])

    def test_corrupt_classifier_keeps_baselines_available(self):
        def change(rows): rows['7']['coef'][0] += 10; return rows
        with self.corrupt('models.json', change):
            with self.assertRaisesRegex(ValueError, '分类器'):
                service.run_task({'method': 'astar_ml'})
            self.assertTrue(service.run_task({'method': 'astar_fixed', 'budget': 128})['result']['valid'])


@requires_artifacts('models.json', 'ga_history.json', 'manifest.json', 'llm_summary.json',
                    'llm_audit.json', 'llm_evidence_readout.json')
class CodeOnlyReadinessTests(TestCase):
    def setUp(self):
        from scripts import prepare_lab03
        self.prepare = prepare_lab03
        clear_verified_inputs()

    def test_current_complete_results_are_ready_without_new_model_calls(self):
        with patch('experiments.lab03.llm.local_runtime', side_effect=AssertionError('no model calls')), \
             patch('experiments.lab03.evaluate.run_benchmark', side_effect=AssertionError('no generation')), \
             patch('experiments.lab03.learning.fit_classifier', side_effect=AssertionError('no fitting')):
            self.assertTrue(self.prepare.results_present())
            self.assertTrue(self.prepare.llm_results_present())

    def test_corrupt_ga_history_is_not_misreported_initialized(self):
        original = self.prepare.read_json
        def changed(path):
            value = original(path)
            if path.name == 'ga_history.json':
                value['7']['history'] = value['7']['history'][:1]
            return value
        with patch.object(self.prepare, 'read_json', side_effect=changed):
            self.assertFalse(self.prepare.results_present())

    def test_bad_learning_metric_is_not_misreported_initialized(self):
        original = self.prepare.read_json
        def changed(path):
            value = original(path)
            if path.name == 'learning_metrics.json': value[0]['f1'] = 'corrupt'
            return value
        with patch.object(self.prepare, 'read_json', side_effect=changed):
            self.assertFalse(self.prepare.results_present())

    def test_readiness_rejects_finite_changed_llm_cache(self):
        original = self.prepare.read_json
        def changed(path):
            value = original(path)
            if path.name == 'llm_cache.json':
                row = value['test-7-0-Q0']; key = next(iter(row['scores']))
                row['scores'][key] = .12345
            return value
        with patch.object(self.prepare, 'read_json', side_effect=changed):
            self.assertFalse(self.prepare.llm_results_present())

    def test_readiness_rejects_summary_without_matching_raw_predictions(self):
        original = self.prepare.read_json
        def changed(path):
            value = original(path)
            if path.name == 'llm_summary.json': value['legal_check_accuracy'] = 1.
            return value
        with patch.object(self.prepare, 'read_json', side_effect=changed):
            self.assertFalse(self.prepare.llm_results_present())

    def test_shared_reader_rejects_duplicate_and_nonfinite_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
                path.write_text(text, encoding='utf-8')
                with self.subTest(text=text), self.assertRaises(ValueError):
                    self.prepare.read_json(path)


class IndependentKernelMatrixTests(TestCase):
    def test_all_nine_methods_match_raw_edge_oracle_on_multigraphs(self):
        # Oracle intentionally avoids Graph.successors/goal/valid_path/enumerate_paths.
        for seed in range(50):
            rng = random.Random(1000 + seed)
            nodes = [Node(str(i), f'N{i}', kind='Person' if i == 0 else 'Company',
                          sector=rng.choice(['A', 'B']), listed=rng.choice([True, False]))
                     for i in range(6)]
            edges = [Edge(str(i), str(rng.randrange(6)), str(rng.randrange(6)),
                          rng.choice(['HOLDS', 'TRADES']), rng.choice([.54, .55, 1.]),
                          rng.choice([True, True, False]), rng.choice([True, True, False]))
                     for i in range(24)]
            graph = Graph(nodes, edges)
            for hops in (2, 3, 4):
                query = Query(f'q-{seed}-{hops}', '0', tuple(rng.choice(['HOLDS', 'TRADES'])
                              for _ in range(hops)), rng.choice(['A', 'B']), rng.choice([True, False]))
                def oracle(allow_bad):
                    found = set(); todo = [(('0',), ())]
                    while todo:
                        ns, es = todo.pop()
                        if len(es) == hops:
                            node = graph.nodes[ns[-1]]
                            if node.kind == 'Company' and node.sector == query.sector and node.listed == query.listed:
                                found.add((ns, es))
                            continue
                        for edge in edges:
                            if (edge.source == ns[-1] and edge.relation == query.relations[len(es)]
                                    and edge.target not in ns and (allow_bad or
                                    edge.verified and edge.active and edge.confidence >= .55)):
                                todo.append((ns + (edge.target,), es + (edge.id,)))
                    return found
                strict, permissive = oracle(False), oracle(True)
                scores = {path_key(s.edges + (e.id,)): rng.random() for s, e in candidate_actions(graph, query)}
                model = {'coef': [rng.uniform(-10, 10) for _ in range(6)],
                         'intercept': rng.uniform(-10, 10), 'seed': seed}
                for method in METHODS:
                    for expert in (True, False):
                        result = search(graph, query, method, 256, weights=FIXED, model=model,
                                        seed=seed, expert=expert, llm_scores=scores, trace=False)
                        expected = permissive if not expert or method == 'hybrid_no_expert' else strict
                        with self.subTest(seed=seed, hops=hops, method=method, expert=expert):
                            self.assertEqual(result['found'], bool(expected))
                            self.assertEqual(result['status'], 'found' if expected else 'exhausted')
                            witness = (tuple(result['path']), tuple(result['edge_ids']))
                            if result['found']: self.assertIn(witness, expected)
                            self.assertEqual(result['valid'], witness in strict)
