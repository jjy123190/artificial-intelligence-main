"""Recompute every published comparison and reject missing, duplicate or stale evidence."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

from .dataset import ROOT, load_dataset, unpack
from .domain import State, digest, path_key, expert_legal
from .search import search, HYBRID_ML_WEIGHT
from .evaluate import summarize, paired_effects, boundary_evaluation
from .learning import classifier_evaluation
from .artifacts import current_manifest, source_hashes, provenance
from .llm import (candidate_actions, bound_scores, prompt_for, legality_prompt_for, audit_explanation,
                  SCORE_PROTOCOL, MODEL_ID, REVISION, WEIGHTS_SHA256, LEGALITY_TOKENS,
                  controlled_explanation_options, controlled_explanation_prompt, EXPLANATION_PROTOCOL)
from .diagnostics import llm_diagnostics, controlled_counterexamples
from .service import valid_model, valid_ga, validate_replay

OUT = ROOT / 'outputs/lab03'
BENCHMARK_METHODS = ('bfs', 'astar_admissible', 'astar_fixed', 'astar_ga', 'astar_ml', 'mcts', 'hybrid', 'hybrid_no_expert')
RESULT_FIELDS = ('valid', 'found', 'status', 'expanded', 'total_work', 'successor_checks', 'heuristic_evaluations', 'heuristic_probe_states', 'proof_pruned', 'proof_checks', 'peak_frontier', 'adjacency_edge_scans', 'heuristic_edge_checks', 'total_edge_checks', 'successor_cache_hits', 'successor_cache_entries', 'hops', 'redundancy_ratio', 'redundant_path_ratio')


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8-sig'))


def converted_rows():
    with (OUT / 'runs.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ('seed', 'budget', 'hops_requested', 'expanded', 'total_work', 'successor_checks', 'heuristic_evaluations', 'heuristic_probe_states', 'proof_pruned', 'proof_checks', 'adjacency_edge_scans', 'heuristic_edge_checks', 'total_edge_checks', 'successor_cache_hits', 'successor_cache_entries', 'expert_rejections'):
            row[key] = int(row[key])
        for key in ('valid', 'found', 'answerable'):
            assert row[key] in ('True', 'False'), (key, row[key])
            row[key] = row[key] == 'True'
        for key in ('redundancy_ratio', 'redundant_path_ratio', 'elapsed_ms'):
            row[key] = float(row[key])
            assert math.isfinite(row[key]) and row[key] >= 0, key
        row['hops'] = int(row['hops']) if row['hops'] else None
        row['peak_frontier'] = int(row['peak_frontier']) if row['peak_frontier'] else None
    return rows


def verify():
    from .calibration import load_readout,verify_readout,probability as readout_probability
    readout=load_readout();readout_verified=verify_readout()
    data = load_dataset()
    models, ga = read('models.json'), read('ga_history.json')
    assert current_manifest(read('manifest.json')), '基准源码或数据指纹失配'
    tasks = {q.id: (g, q, r['seed']) for r in data for g, qs in [unpack(r)] for q in qs}
    seen_nodes, seen_queries = set(), set()
    for record in data:
        graph, queries = unpack(record)
        assert not seen_nodes.intersection(graph.nodes), '图划分共享实体'
        seen_nodes.update(graph.nodes)
        for query in queries:
            assert query.id not in seen_queries, '重复问题编号'
            seen_queries.add(query.id)
    for seed in (7, 19, 43):
        model, optimized = models[str(seed)], ga[str(seed)]
        assert valid_model(model, seed) and valid_ga(optimized, seed), '学习产物无效或过期'
        assert set(model['train_graph_ids']) == {r['graph']['id'] for r in data if r['split'] == 'train' and r['seed'] == seed}
        assert set(optimized['validation_graph_ids']) == {r['graph']['id'] for r in data if r['split'] == 'validation' and r['seed'] == seed}
        assert [h['generation'] for h in optimized['history']] == list(range(optimized['generations'] + 1))
        assert optimized['weights'] == optimized['history'][-1]['weights']
        validation = [(g, q) for g, q, s in tasks.values() if g.split == 'validation' and s == seed]
        for generation in optimized['history']:
            runs = [search(g, q, 'astar_ga', optimized['budget'], weights=generation['weights'], seed=seed, trace=False) for g, q in validation]
            success = sum(r['valid'] for r in runs) / len(runs)
            work = sum(r['expanded'] for r in runs) / (len(runs) * optimized['budget'])
            assert math.isclose(success, generation['success_rate'], abs_tol=1e-12)
            assert math.isclose(work, generation['expanded_fraction'], abs_tol=1e-12)
            assert math.isclose(success - .12 * work, generation['fitness'], abs_tol=1e-12)
        values = [h['fitness'] for h in optimized['history']]
        assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
    rows = converted_rows()
    expected = {(q.id, b, m) for g, q, _ in tasks.values() if g.split in ('test', 'ood') for b in (4, 8, 16) for m in BENCHMARK_METHODS}
    keys = [(r['query_id'], r['budget'], r['method']) for r in rows]
    assert len(keys) == len(set(keys)) and set(keys) == expected, '基准组合缺失、重复或多余'
    assert read('manifest.json')['runs'] == len(rows)
    for row in rows:
        graph, query, seed = tasks[row['query_id']]
        assert row['seed'] == seed and row['graph_id'] == graph.id and row['split'] == graph.split
        assert row['hops_requested'] == query.hops
        w = ga[str(seed)]['weights'][:]
        w[5] = HYBRID_ML_WEIGHT
        actual = search(graph, query, row['method'], row['budget'], w, models[str(seed)], seed, trace=False)
        for key in RESULT_FIELDS + ('expert_rejections',):
            assert actual[key] == row[key], (query.id, key)
    summary = summarize(rows)
    assert summary == read('summary.json'), '汇总表不一致'
    with (OUT / 'summary.csv').open(encoding='utf-8-sig', newline='') as f:
        csv_summary = list(csv.DictReader(f))
    assert len(csv_summary) == len(summary)
    for saved, computed in zip(csv_summary, summary):
        assert saved == {k: '' if v is None else str(v) for k, v in computed.items()}, 'CSV统计表不一致'
    assert paired_effects(rows) == read('paired_effects.json'), '配对区间无法复算'
    learning = [{'seed': seed, **classifier_evaluation(data, models[str(seed)])} for seed in (7, 19, 43)]
    assert learning == read('learning_metrics.json'), '分类器评估无法复算'
    from .domain import enumerate_paths
    for failure in read('failure_cases.json'):
        graph, query, seed = tasks[failure['query']['id']]
        w = ga[str(seed)]['weights'][:]; w[5] = HYBRID_ML_WEIGHT
        actual = search(graph, query, failure['method'], failure['result']['budget'], w, models[str(seed)], seed, trace=False)
        assert not actual['valid'] and failure['oracle_paths'] == len(enumerate_paths(graph, query))
        for key in RESULT_FIELDS:
            assert actual[key] == failure['result'][key], ('failure', key)
    traces = read('search_examples.json')
    for example in traces:
        graph, query, seed = tasks[example['query']['id']]
        assert example['graph'] == graph.to_dict()
        w = ga[str(seed)]['weights'][:]; w[5] = HYBRID_ML_WEIGHT
        actual = search(graph, query, example['result']['method'], example['result']['budget'], w, models[str(seed)], seed)
        assert actual['trace'] == example['result']['trace'], '搜索树回放不一致'
    boundary = boundary_evaluation()
    assert boundary == read('boundary_cases.json') and boundary['passed'] == boundary['total']
    cache, audit, llm_rows = read('llm_cache.json'), read('llm_audit.json'), read('llm_runs.json')
    selected = {q.id for r in data if r['split'] == 'test' and r['graph']['id'] == f'test-{r["seed"]}-0' for g, qs in [unpack(r)] for q in qs[:3]}
    assert set(cache) == selected and {a['query_id'] for a in audit} == selected and len(audit) == len(selected)
    for item in audit:
        graph, query, _ = tasks[item['query_id']]
        scores = validate_replay(cache[query.id], graph, query)
        candidates = candidate_actions(graph, query)
        assert len(item['evaluations']) == len(candidates)
        for record, (state, edge) in zip(item['evaluations'], candidates):
            prompt = prompt_for(graph, query, edge, state)
            assert record['prompt'] == prompt and record['prompt_sha256'] == digest(prompt), 'LLM提示词来源失配'
            assert record['legality_prompt'] == legality_prompt_for(prompt), '合法性提示词来源失配'
            assert record['model'] == MODEL_ID and record['revision'] == REVISION and record['weights_sha256'] == WEIGHTS_SHA256
            assert record['edge_id'] == edge.id and record['path_key'] == path_key(state.edges + (edge.id,))
            assert record['path_edges'] == list(state.edges + (edge.id,)) and record['parent_edges'] == list(state.edges)
            assert record['expert_legal'] == all(expert_legal(graph.edges[e]) for e in state.edges + (edge.id,))
            assert record['protocol'] == SCORE_PROTOCOL and record['inference_ms'] >= 0
            assert set(record['probabilities']) == {'A', 'B', 'C'} and set(record['legality_probabilities']) == set(LEGALITY_TOKENS)
            p, legal = record['probabilities'], record['legality_probabilities']
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in [*p.values(), *legal.values()])
            assert math.isclose(sum(p.values()), 1, abs_tol=1e-6) and math.isclose(sum(legal.values()), 1, abs_tol=1e-6)
            assert record['score'] == p['A'] + .5 * p['B'] and record['legal'] == (legal['True'] >= legal['False'])
            assert record['readout_sha256']==digest(readout)
            value=readout_probability(readout,record['evidence_hidden'])
            assert math.isclose(value,record['calibrated_confidence'],abs_tol=1e-7)
            assert record['calibrated_legal']==(value>=.5)
        assert bound_scores(item['evaluations']) == scores == item['scores']
        assert math.isclose(cache[query.id]['llm_ms'], sum(e['inference_ms'] for e in item['evaluations']))
    assert {(r['query_id'], r['method']) for r in llm_rows} == {(q, m) for q in selected for m in ('hybrid', 'hybrid_llm')}
    assert len(llm_rows) == 2 * len(selected)
    for row in llm_rows:
        graph, query, seed = tasks[row['query_id']]
        assert row['seed'] == seed and row['graph_id'] == graph.id and row['budget'] == 8
        w = ga[str(seed)]['weights'][:]; w[5] = HYBRID_ML_WEIGHT
        actual = search(graph, query, row['method'], 8, w, models[str(seed)], seed, llm_scores=cache[query.id]['scores'])
        for key in RESULT_FIELDS:
            assert actual[key] == row[key], (query.id, key)
        assert row['llm_ms'] == (cache[query.id]['llm_ms'] if row['method'] == 'hybrid_llm' else 0.)
    llm_summary = read('llm_summary.json')
    assert llm_summary['source_sha256'] == source_hashes() and llm_summary['dataset_sha256'] == provenance()['dataset_sha256']
    assert llm_summary['results'] == summarize(llm_rows) and llm_summary['total_inference_ms'] == sum(r['llm_ms'] for r in llm_rows)
    records = [e for a in audit for e in a['evaluations']]
    assert llm_summary['prediction_cache_hits']==sum(r['prediction_cache_hits'] for r in records)
    assert llm_summary['processed_prompt_tokens']==sum(r['processed_prompt_tokens'] for r in records)
    assert all(type(r['prediction_cache_hits']) is int and 0<=r['prediction_cache_hits']<=2
               and 0<=r['processed_prompt_tokens']<=r['prompt_tokens'] for r in records)
    assert llm_summary['llm_preparation_edge_scans']==sum(r['llm_preparation_edge_scans'] for r in llm_rows)
    assert llm_summary['llm_preparation_states']==sum(r['llm_preparation_states'] for r in llm_rows)
    assert llm_summary['legal_checks'] == len(records)
    assert llm_summary['legal_check_accuracy'] == sum(e['legal'] == e['expert_legal'] for e in records) / len(records)
    assert llm_summary['model'] == MODEL_ID and llm_summary['revision'] == REVISION and llm_summary['weights_sha256'] == WEIGHTS_SHA256
    diagnostics=llm_diagnostics(audit,llm_rows)
    assert diagnostics == read('llm_diagnostics.json'), 'LLM错误分析无法复算'
    assert llm_summary['legal_metrics'] == diagnostics['legality']
    assert llm_summary['calibrated_legal_metrics']==diagnostics['calibrated_legality']
    assert llm_summary['readout_sha256']==digest(readout)
    assert llm_summary['paired_diagnostics'] == {k:diagnostics[k] for k in llm_summary['paired_diagnostics']}
    explanations = read('llm_explanations.json')
    for explanation in explanations:
        graph, query, _ = tasks[explanation['query_id']]
        state = State(tuple(explanation['path']), tuple(explanation['edge_ids']))
        assert graph.valid_path(state, query), '解释绑定了无效路径'
        result = {'valid': True, 'path': explanation['path'], 'edge_ids': explanation['edge_ids']}
        checked = audit_explanation(graph, query, result, explanation['text'])
        assert all(explanation[k] == v for k, v in checked.items()), '解释认证状态失配'
        options=controlled_explanation_options(graph,query,result)
        prompt=controlled_explanation_prompt(options)
        assert explanation['controlled_protocol']==EXPLANATION_PROTOCOL
        assert explanation['controlled_prompt']==prompt and explanation['controlled_prompt_sha256']==digest(prompt)
        probabilities=explanation['controlled_probabilities']
        assert set(probabilities)==set('ABC') and all(math.isfinite(p) and 0<=p<=1 for p in probabilities.values())
        assert math.isclose(sum(probabilities.values()),1,abs_tol=1e-6)
        choice=max('ABC',key=lambda k:probabilities[k])
        assert explanation['controlled_choice']==choice and explanation['controlled_text']==options['ABC'.index(choice)]
        assert explanation['controlled_grounded'] is True

    assert llm_summary['explanations'] == len(explanations) and llm_summary['explanation_grounded'] == sum(e['grounded'] for e in explanations)
    assert llm_summary['controlled_explanations']==len(explanations)
    stress = read('hallucination_stress.json')
    assert stress['protocol'] == SCORE_PROTOCOL and {r['query_id'] for r in stress['cases']} == selected
    for row in stress['cases']:
        graph, query, seed = tasks[row['query_id']]
        w = ga[str(seed)]['weights'][:]; w[5] = HYBRID_ML_WEIGHT
        corrupt = {path_key(st.edges + (e.id,)): 1. if not all(expert_legal(graph.edges[x]) for x in st.edges + (e.id,)) else 0.
                   for st, e in candidate_actions(graph, query)}
        actual = search(graph, query, 'hybrid_llm', 16, w, models[str(seed)], seed, llm_scores=corrupt)
        assert all(row[k] == actual[k] for k in ('valid', 'found', 'expert_rejections'))
        unprotected=search(graph,query,'hybrid_llm',16,w,models[str(seed)],seed,expert=False,llm_scores=corrupt)
        assert row['without_expert']=={k:unprotected[k] for k in ['valid','found','status','expanded','expert_rejections']}
    assert stress['invalid_answers'] == sum(r['found'] and not r['valid'] for r in stress['cases'])
    assert stress['without_expert_invalid_answers'] == sum(r['without_expert']['found'] and not r['without_expert']['valid'] for r in stress['cases'])
    assert stress['counterexamples'] == controlled_counterexamples(), '错误评分机制反例无法复算'
    from .optimization import audit_optimization
    optimization = audit_optimization()
    assert optimization == read('optimization/comparison.json'), '优化前后对比无法复算'
    from .refinement import audit_refinement, confirm
    refinement=audit_refinement()
    assert refinement==read('refinement/comparison.json'), '本轮细化对比无法复算'
    confirmation=confirm(write=False)
    assert confirmation==read('refinement/confirmation.json'), '新种子确认试验无法复算'
    from .structure import audit_structure, confirm as confirm_structure
    structure=audit_structure()
    assert structure==read('structure/comparison.json'), '第三阶段对比无法复算'
    fresh=confirm_structure(write_result=False)
    assert fresh==read('structure/confirmation.json'), '第三阶段新种子确认无法复算'
    from .posthoc import analyze_structural_failures
    failures=analyze_structural_failures(write_result=False)
    assert failures==read('structure/failure_analysis.json'), '确认集失败轨迹无法复算'
    from .reliability import audit_reliability,confirm_search
    reliability=audit_reliability()
    assert reliability==read('reliability/comparison.json'), '第四阶段比较无法复算'
    newest=confirm_search(write_result=False)
    assert newest==read('reliability/confirmation.json'), '第四阶段新确认无法复算'
    from .verification_reliability import verify_llm_confirmation
    llm_confirm=verify_llm_confirmation()
    from .service import snapshot as live_snapshot
    served=live_snapshot()
    assert not served['warnings'], ('当前网页证据链无法校验', served['warnings'])
    assert served['summary']==summary and served['llm_summary']==llm_summary
    assert served['llm_confirmation']['prefixes']==llm_confirm['fresh_prefixes']
    return {'runtime_evidence_chain':'matched','readout':readout_verified,'fourth_stage_pairs':reliability['pairs'],
            'fourth_stage_new_pairs':newest['paired_runs'],'fourth_stage_negative_pairs':newest['negative_pairs'],
            'fourth_stage_llm':llm_confirm,'posthoc_failure_cases':failures['count'],'fresh_confirmation_pairs':fresh['paired_runs'],'fresh_confirmation_questions':fresh['question_count'],
            'historical_confirmation_pairs':confirmation['paired_runs'],'structure_pairs_rechecked':structure['pairs'],'refinement_pairs_rechecked':refinement['pairs'],'optimization_pairs_rechecked': optimization['same_question_pairs'], 'status': 'passed', 'benchmark_runs_replayed': len(rows), 'llm_paired_runs_replayed': len(llm_rows),
            'tables_recomputed': len(summary), 'paired_intervals_recomputed': len(read('paired_effects.json')),
            'ga_generations_recomputed': sum(len(g['history']) for g in ga.values()), 'classifiers_rechecked': len(learning),
            'boundary_runs': boundary['total'], 'traces_replayed': len(traces), 'llm_prompts_and_explanations': 'matched',
            'dataset_and_core_hashes': 'matched', 'llm_diagnostics': 'matched',
            'controlled_counterexamples': 2, 'stress_paired_runs': 2*len(stress['cases'])}


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
