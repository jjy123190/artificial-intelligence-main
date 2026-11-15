"""Request-local reuse and independent validation of records served by the UI.

No model inference, training or new benchmark takes place here. This verifies
saved records against their raw probabilities, paths and deterministic replay.
It detects incomplete/corrupt artifacts, not coordinated forgery of all evidence.
"""
from __future__ import annotations

from contextvars import ContextVar
from collections import OrderedDict
from threading import Lock
import hashlib
import marshal
from functools import wraps
import math

from .domain import digest, expert_legal, path_key
from .llm import (MODEL_ID, REVISION, WEIGHTS_SHA256, SCORE_PROTOCOL,
                  LEGALITY_TOKENS, EXPLANATION_PROTOCOL, candidate_actions,
                  prompt_for, legality_prompt_for, bound_scores, audit_explanation,
                  controlled_explanation_options, controlled_explanation_prompt)
from .calibration import probability, READOUT_PROTOCOL
from .diagnostics import legality_metrics, llm_diagnostics, controlled_counterexamples
from .search import search, HYBRID_ML_WEIGHT

_request_values = ContextVar('lab03_request_values', default=None)

_verified_inputs = OrderedDict()
_verdict_lock = Lock()
VERDICT_CACHE_LIMIT = 16


def clear_verified_inputs():
    with _verdict_lock:
        _verified_inputs.clear()


def verified_inputs(kind, inputs, check):
    """Cache only a successful check token, never data, failures or model output.

    Every request still reads the source files and hashes all current values
    needed by the check. Edited content (including same-size/mtime edits) misses
    the cache. It is bounded, thread-safe and stores only digests, not the large
    raw records. Duplicate concurrent cold checks are safe.
    """
    # Inputs are in-memory JSON builtins. Binary serialization is used ONLY
    # for a process-local digest: never persist or unmarshal untrusted bytes.
    # It preserves bool/int/float distinctions without reformatting 300,000
    # floating-point activations as decimal strings on every request.
    try:
        blob = marshal.dumps(inputs, 4)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('实验记录包含不支持的非JSON类型。') from exc
    key = (kind, hashlib.sha256(blob).hexdigest())
    with _verdict_lock:
        if key in _verified_inputs:
            _verified_inputs.move_to_end(key)
            return
    check()
    with _verdict_lock:
        _verified_inputs[key] = True
        _verified_inputs.move_to_end(key)
        while len(_verified_inputs) > VERDICT_CACHE_LIMIT:
            _verified_inputs.popitem(last=False)


def request_scoped(function):
    """Share work only within this call/context; a later request rereads files."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _request_values.get() is not None:
            return function(*args, **kwargs)
        token = _request_values.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _request_values.reset(token)
    return wrapped


def once(key, factory):
    values = _request_values.get()
    if values is None:
        return factory()
    if key not in values:
        try:
            values[key] = (True, factory())
        except ValueError as exc:
            # Missing/corrupt artifacts must not trigger nine expensive retries.
            values[key] = (False, exc)
    success, value = values[key]
    if not success:
        raise value
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def count(value, maximum=None):
    return type(value) is int and value >= 0 and (maximum is None or value <= maximum)


def same(actual, expected):
    """Booleans must not pass as the numbers 0 and 1 in JSON evidence."""
    if isinstance(expected, bool):
        return type(actual) is bool and actual is expected
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(same(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, (tuple, list)):
        return (isinstance(actual, (tuple, list)) and len(actual) == len(expected)
                and all(same(a, b) for a, b in zip(actual, expected)))
    if type(expected) in (int, float):
        return finite(actual) and actual == expected
    return actual == expected


def checked_shape(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
            raise ValueError('实验三原始证据结构不完整或类型不正确；请重新生成相关记录。') from exc
    return wrapped


def identity(row):
    require(isinstance(row, dict) and row.get('model') == MODEL_ID
            and row.get('revision') == REVISION and row.get('weights_sha256') == WEIGHTS_SHA256,
            '模型记录与固定Qwen身份不一致。')


def probabilities(values, labels):
    require(isinstance(values, dict) and set(values) == set(labels)
            and all(finite(v) and 0 <= v <= 1 for v in values.values())
            and math.isclose(sum(values.values()), 1, abs_tol=1e-6),
            '模型原始概率不完整、非有限或未归一化。')


def close(actual, expected, tolerance=1e-7):
    return finite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=tolerance)


@checked_shape
def prediction_records(records, graph, query, readout, readout_sha):
    preparation = []
    candidates = candidate_actions(graph, query, account=preparation.append)
    require(isinstance(records, list) and len(records) == len(candidates), 'LLM前缀原始记录缺失或重复。')
    for row, (state, edge) in zip(records, candidates):
        identity(row)
        prompt = prompt_for(graph, query, edge, state)
        prefix = state.edges + (edge.id,)
        expected = {'prompt': prompt, 'prompt_sha256': digest(prompt),
                    'legality_prompt': legality_prompt_for(prompt), 'legality_tokens': list(LEGALITY_TOKENS),
                    'edge_id': edge.id, 'path_edges': list(prefix), 'parent_edges': list(state.edges),
                    'path_key': path_key(prefix), 'protocol': SCORE_PROTOCOL, 'backend': 'local',
                    'readout_protocol': READOUT_PROTOCOL, 'readout_sha256': readout_sha,
                    'expert_legal': all(expert_legal(graph.edges[e]) for e in prefix)}
        require(all(same(row.get(k), v) for k, v in expected.items()), 'LLM提示、路径、协议或评估标签与当前图不一致。')
        raw, value = row['legality_probabilities'], row['probabilities']
        probabilities(raw, LEGALITY_TOKENS)
        probabilities(value, 'ABC')
        require(close(row['score'], value['A'] + .5 * value['B'])
                and close(row['legal_confidence'], raw['True'])
                and same(row['legal'], raw['True'] >= raw['False'])
                and row['label'] == max((value[k], k) for k in 'ABC')[1],
                'LLM评分或合法性与保存的原始概率不一致。')
        hidden = row['evidence_hidden']
        require(isinstance(hidden, list) and len(hidden) == readout['hidden_dim']
                and all(finite(v) for v in hidden), 'LLM证据隐藏向量不合法。')
        calibrated = probability(readout, hidden)
        require(close(row['calibrated_confidence'], calibrated) and same(row['calibrated_legal'], calibrated >= .5),
                'LLM分类头预测与保存隐藏表示不一致。')
        require(finite(row['inference_ms']) and row['inference_ms'] >= 0
                and count(row['prompt_tokens']) and count(row['processed_prompt_tokens'], row['prompt_tokens'])
                and count(row['prediction_cache_hits'], 2), 'LLM推断成本记录不合法。')
    return {'scores': bound_scores(records), 'scored_edges': len(records),
            'llm_ms': sum(r['inference_ms'] for r in records),
            'preparation_edge_scans': sum(preparation), 'preparation_states': len(preparation)}


@checked_shape
def audit_index(audit):
    require(isinstance(audit, list), '缺少LLM原始推断审计，不能仅凭缓存评分重放。')
    values = {}
    for item in audit:
        qid = item['query_id']
        require(isinstance(qid, str) and qid not in values, 'LLM原始推断审计存在重复问题。')
        values[qid] = item
    return values


@checked_shape
def original_audit(item, graph, query, readout, readout_sha):
    require(item is not None and item['graph_id'] == graph.id and item['query_id'] == query.id
            and digest(item['query']) == digest(query.__dict__), '缺少与问题绑定的LLM原始推断记录。')
    values = prediction_records(item['evaluations'], graph, query, readout, readout_sha)
    require(same(item['scores'], values['scores']), 'LLM审计内评分与原始概率不一致。')
    return values


RESULT_FIELDS = ('valid', 'found', 'status', 'expanded', 'total_work', 'successor_checks',
                 'heuristic_evaluations', 'heuristic_probe_states', 'proof_pruned', 'proof_checks',
                 'peak_frontier', 'adjacency_edge_scans', 'heuristic_edge_checks', 'total_edge_checks',
                 'successor_cache_hits', 'successor_cache_entries', 'hops', 'redundancy_ratio', 'redundant_path_ratio')


@checked_shape
def original_llm_summary(summary, audit, runs, explanations, diagnostics, stress,
                         task_map, caches, models, ga, readout, readout_sha, hashes, provenance):
    """18 deterministic search replays are cheap; never run Qwen in a web request."""
    from .evaluate import summarize
    selected = {qid: entry for qid, entry in task_map.items() if qid.startswith('test-')}
    indexed = audit_index(audit)
    require(set(indexed) == set(selected), 'LLM审计问题集与预定实测子集不一致。')
    require(isinstance(runs, list) and len(runs) == 2 * len(selected), 'LLM搜索配对记录不完整。')
    expected_keys = {(qid, method) for qid in selected for method in ('hybrid', 'hybrid_llm')}
    require({(r['query_id'], r['method']) for r in runs} == expected_keys, 'LLM搜索配对问题缺失或重复。')
    solved = {}
    for row in runs:
        graph, query, seed = selected[row['query_id']]
        cache = caches[query.id]
        require(row['seed'] == seed and row['graph_id'] == graph.id and row['budget'] == 8
                and row['split'] == 'test_llm_subset' and finite(row['elapsed_ms']) and row['elapsed_ms'] >= 0,
                'LLM搜索问题、预算或成本不一致。')
        weights = ga[str(seed)]['weights'][:]
        weights[5] = HYBRID_ML_WEIGHT
        result = search(graph, query, row['method'], 8, weights, models[str(seed)], seed,
                        llm_scores=cache['scores'], trace=False)
        require(all(same(row.get(k), result[k]) for k in RESULT_FIELDS), 'LLM搜索记录与当前确定性重放不一致。')
        for key, cache_key in [('llm_ms', 'llm_ms'), ('llm_preparation_edge_scans', 'preparation_edge_scans'),
                               ('llm_preparation_states', 'preparation_states'), ('scored_edges', 'scored_edges')]:
            expected = cache[cache_key] if row['method'] == 'hybrid_llm' else 0
            require(same(row[key], expected), 'LLM配对成本与原始推断记录不一致。')
        if row['method'] == 'hybrid_llm' and result['valid']:
            solved[query.id] = result
    require(isinstance(explanations, list), '模型解释记录不完整。')
    expected_explanations = [qid for qid in selected if qid in solved][:3]
    require([e['query_id'] for e in explanations] == expected_explanations, '模型解释问题缺失、重复或次序不符。')
    for row in explanations:
        graph, query, _ = selected[row['query_id']]
        result = solved[query.id]
        identity(row)
        require(row['graph_id'] == graph.id and row['graph_sha256'] == digest(graph.to_dict())
                and row['query_sha256'] == digest(query.__dict__) and same(row['path'], result['path'])
                and same(row['edge_ids'], result['edge_ids']) and isinstance(row['text'], str),
                '模型解释与真实搜索路径不一致。')
        checked = audit_explanation(graph, query, result, row['text'])
        require(all(same(row.get(k), v) for k, v in checked.items()), '模型自由解释的证据认证标记不一致。')
        options = controlled_explanation_options(graph, query, result)
        prompt = controlled_explanation_prompt(options)
        ps = row['controlled_probabilities']
        probabilities(ps, 'ABC')
        choice = max('ABC', key=lambda k: ps[k])
        require(row['controlled_protocol'] == EXPLANATION_PROTOCOL and row['controlled_prompt'] == prompt
                and row['controlled_prompt_sha256'] == digest(prompt) and row['controlled_choice'] == choice
                and row['controlled_text'] == options['ABC'.index(choice)]
                and row['controlled_grounded'] is True and row['controlled_generated'] is True,
                '受控解释不是模型选择的已核验证据措辞。')
    computed = llm_diagnostics(audit, runs)
    require(same(diagnostics, computed), 'LLM错误分析与原始预测不一致。')
    records = [r for a in audit for r in a['evaluations']]
    identity(summary)
    expected = {'protocol': SCORE_PROTOCOL, 'source_sha256': hashes, 'dataset_sha256': provenance['dataset_sha256'],
                'readout_protocol': READOUT_PROTOCOL, 'readout_sha256': readout_sha,
                'readout_training': {k: readout[k] for k in ('training_prefixes', 'trained_unique_prompts',
                    'training_elapsed_ms', 'training_model_passes', 'training_prompt_tokens', 'l2')},
                'questions': len(selected), 'results': summarize(runs),
                'total_inference_ms': sum(r['llm_ms'] for r in runs),
                'llm_preparation_edge_scans': sum(r['llm_preparation_edge_scans'] for r in runs),
                'llm_preparation_states': sum(r['llm_preparation_states'] for r in runs),
                'legal_checks': len(records), 'legal_check_accuracy': computed['legality']['accuracy'],
                'legal_metrics': computed['legality'], 'calibrated_legal_metrics': computed['calibrated_legality'],
                'paired_diagnostics': {k: computed[k] for k in ('improved_answers', 'lost_answers', 'extra_expansion_cases',
                                                              'fewer_expansion_cases', 'extra_expansions')},
                'prediction_cache_hits': sum(r['prediction_cache_hits'] for r in records),
                'processed_prompt_tokens': sum(r['processed_prompt_tokens'] for r in records),
                'controlled_explanations': sum(e['controlled_grounded'] for e in explanations),
                'explanation_grounded': sum(e['grounded'] for e in explanations), 'explanations': len(explanations)}
    require(all(same(summary.get(k), v) for k, v in expected.items()), 'LLM展示汇总与逐条实测记录不一致；旧统计已隐藏。')
    require(stress['protocol'] == SCORE_PROTOCOL and isinstance(stress['cases'], list)
            and [r['query_id'] for r in stress['cases']] == list(selected), '受控错误注入记录不完整。')
    for row in stress['cases']:
        graph, query, seed = selected[row['query_id']]
        weights = ga[str(seed)]['weights'][:]; weights[5] = HYBRID_ML_WEIGHT
        scores = {path_key(s.edges + (e.id,)): (1. if not all(expert_legal(graph.edges[x])
                   for x in s.edges + (e.id,)) else 0.) for s, e in candidate_actions(graph, query)}
        for expert in (True, False):
            result = search(graph, query, 'hybrid_llm', 16, weights, models[str(seed)], seed,
                            expert=expert, llm_scores=scores, trace=False)
            record = row if expert else row['without_expert']
            keys = ('valid', 'found', 'expert_rejections') if expert else ('valid', 'found', 'status', 'expanded', 'expert_rejections')
            require(all(same(record[k], result[k]) for k in keys), '受控错误注入记录与重放不一致。')
    require(stress['invalid_answers'] == sum(r['found'] and not r['valid'] for r in stress['cases'])
            and stress['without_expert_invalid_answers'] == sum(r['without_expert']['found'] and not r['without_expert']['valid'] for r in stress['cases'])
            and same(stress['counterexamples'], controlled_counterexamples()), '受控错误注入统计不一致。')
    return True


@checked_shape
def confirmation_rows(record):
    """Check full matrix coverage and recompute displayed aggregates, not new runs."""
    from .structure import paired_summary, regressions
    graphs = record['graphs']
    expected_graphs = {f'{split}-{seed}-{index}': split for split in
                       ('reliability_confirmation', 'reliability_confirmation_shift')
                       for seed in (5039, 7001, 9001) for index in range(4)}
    require(len(graphs) == len(expected_graphs) and {g['id']: g['split'] for g in graphs} == expected_graphs
            and all(g['questions'] == 9 for g in graphs), '新图确认图集不完整。')
    qids = {g + f'-Q{i}': (g, split) for g, split in expected_graphs.items() for i in range(9)}
    methods = ('bfs', 'astar_admissible', 'astar_fixed', 'astar_ga', 'astar_ml', 'mcts', 'hybrid', 'hybrid_no_expert')
    for key, budgets, variants in [('rows', (4, 8, 16), methods), ('negative_rows', (8, 64), ('astar_fixed', 'hybrid'))]:
        raw = record[key]
        negative = key == 'negative_rows'
        expected = {(qid + ('-negative' if negative else ''), budget, m) for qid in qids for budget in budgets for m in variants}
        require(isinstance(raw, list) and len(raw) == len(expected)
                and {(r['query_id'], r['budget'], r['method']) for r in raw} == expected, '新图确认逐题矩阵缺失或重复。')
        for row in raw:
            qid = row['query_id'].removesuffix('-negative') if negative else row['query_id']
            require((row['graph_id'], row['split']) == qids[qid], '新图确认问题和图不匹配。')
            for side in ('before', 'after'):
                value = row[side]
                require(type(value['valid']) is bool and type(value['found']) is bool
                        and (not value['valid'] or value['found'])
                        and value['status'] in ('found', 'exhausted', 'budget_exhausted')
                        and value['found'] == (value['status'] == 'found')
                        and count(value['expanded'], row['budget']) and count(value['total_work'])
                        and value['total_work'] >= value['expanded'] and count(value['total_edge_checks'])
                        and (value['hops'] in (2, 3, 4) if value['found'] else value['hops'] is None),
                        '新图确认逐题结果类型、状态或成本不合法。')
    require(same(record['summary'], paired_summary(record['rows']))
            and same(record['regressions'], regressions(record['rows'])), '新图可答统计与逐题结果不一致。')
    groups = {}
    for row in record['negative_rows']:
        groups.setdefault((row['split'], row['budget'], row['method']), []).append(row)
    computed = []
    for (split, budget, method), rows in groups.items():
        computed.append({'split': split, 'budget': budget, 'method': method, 'n': len(rows),
                         **{side: {'correct_exhausted': sum(r[side]['status'] == 'exhausted' and not r[side]['found'] for r in rows),
                                   'false_found': sum(r[side]['found'] for r in rows),
                                   'mean_expanded': sum(r[side]['expanded'] for r in rows) / len(rows),
                                   'mean_edge_checks': sum(r[side]['total_edge_checks'] for r in rows) / len(rows)}
                            for side in ('before', 'after')}})
    require(same(record['negative_summary'], computed), '无解确认统计与逐题结果不一致。')
    return True


@checked_shape
def fresh_llm(record, readout, readout_sha, models, ga, training):
    """Validate all fresh display evidence, including novel and precision subsets."""
    from .reliability import fresh_tasks, numeric_challenge
    tasks = once('fresh_tasks', lambda: {q.id: (g, q, model_seed)
                 for _, model_seed, index, g, queries in fresh_tasks() if index == 0 for q in queries[:3]})
    audit = record['audit']
    require(isinstance(audit, list) and len(audit) == len(tasks)
            and {r['query_id'] for r in audit} == set(tasks), '新LLM确认问题缺失或重复。')
    paired, records = [], []
    for task in audit:
        graph, query, seed = tasks[task['query_id']]
        require(task['graph_id'] == graph.id and task['graph_sha256'] == digest(graph.to_dict())
                and task['query_sha256'] == digest(query.__dict__), '新LLM确认图或问题指纹不一致。')
        rows = task['records']
        checked = prediction_records(rows, graph, query, readout, readout_sha)
        raw_scores = {r['path_key']: r['score'] if r['legal'] else .25*r['score'] for r in rows}
        weights = ga[str(seed)]['weights'][:]; weights[5] = HYBRID_ML_WEIGHT
        for variant, method, scores in [('no_llm', 'hybrid', None), ('raw', 'hybrid_llm', raw_scores),
                                        ('calibrated', 'hybrid_llm', checked['scores'])]:
            result = search(graph, query, method, 8, weights, models[str(seed)], seed, llm_scores=scores, trace=False)
            paired.append({'query_id': query.id, 'variant': variant, 'budget': 8,
                           **{k: result[k] for k in ('valid', 'found', 'status', 'expanded', 'total_edge_checks', 'proof_pruned')}})
        records.extend(rows)
    require(same(record['paired_search'], paired), '新LLM三种搜索配对与当前重放不一致。')
    summary = [{'variant': variant, 'questions': sum(r['variant'] == variant for r in paired),
                'valid': sum(r['valid'] for r in paired if r['variant'] == variant),
                'expanded': sum(r['expanded'] for r in paired if r['variant'] == variant)}
               for variant in ('no_llm', 'raw', 'calibrated')]
    require(same(record['paired_summary'], summary) and same(record['inference_ms'], sum(r['inference_ms'] for r in records)),
            '新LLM搜索汇总或推断时间不一致。')
    raw, head = legality_metrics(records), legality_metrics([{**r, 'legal': r['calibrated_legal']} for r in records])
    require(same(record['raw_metrics'], raw) and same(record['calibrated_metrics'], head), '新LLM合法性统计与原始预测不一致。')
    require(training['protocol'] == READOUT_PROTOCOL, '训练提示来源协议不一致。')
    known_prompts = [r['prompt'] for r in training['unique_prompts']]
    require(len(known_prompts) == readout['trained_unique_prompts'] and len(set(known_prompts)) == len(known_prompts)
            and digest(known_prompts) == readout['training_prompt_sha256'], '未见提示统计的训练来源不一致。')
    known = set(known_prompts)
    novel = [r for r in records if r['legality_prompt'] not in known]
    require(record['novel_prompt_prefixes'] == len(novel) and same(record['novel_raw_metrics'], legality_metrics(novel))
            and same(record['novel_calibrated_metrics'], legality_metrics([{**r, 'legal': r['calibrated_legal']} for r in novel])),
            '新LLM未见提示子集与真实训练提示不一致。')
    numeric = record['numeric_challenge']
    expected = list(numeric_challenge())
    require(isinstance(numeric, list) and len(numeric) == len(expected), '数值精度挑战缺失或重复。')
    for row, facts in zip(numeric, expected):
        require(all(same(row.get(k), v) for k, v in facts.items()), '数值精度挑战不是冻结的问题与独立标签。')
        ps = row['raw_probabilities']
        require(isinstance(ps, list) and len(ps) == 2, '数值精度原始概率数量不符。')
        probabilities(dict(zip(LEGALITY_TOKENS, ps)), LEGALITY_TOKENS)
        hidden = row['hidden']
        require(isinstance(hidden, list) and len(hidden) == readout['hidden_dim'] and all(finite(v) for v in hidden),
                '数值精度隐藏向量不合法。')
        p = probability(readout, hidden)
        require(same(row['raw_legal'], ps[0] >= ps[1]) and close(row['calibrated_probability'], p)
                and same(row['calibrated_legal'], p >= .5), '数值挑战分类结果与概率或学习参数不一致。')
        require(finite(row['elapsed_ms']) and row['elapsed_ms'] >= 0 and count(row['prompt_tokens'])
                and type(row['cache_hit']) is bool, '数值挑战成本记录不合法。')
    for key, source in [('numeric_raw', 'raw_legal'), ('numeric_calibrated', 'calibrated_legal')]:
        expected = legality_metrics([{'legal': r[source], 'expert_legal': r['target']} for r in numeric])
        require(same(record[key], expected), '数值挑战混淆矩阵与逐条预测不一致。')
    return True


@checked_shape
def benchmark_tables(text, summary, paired, failures, manifest):
    """Rebuild the published table/intervals from the complete CSV matrix.

    This is not an independent rerun of all 3,888 searches; full verification
    remains the reproducibility gate. Counts and paired graph bootstraps cannot
    be changed independently of the saved per-question observations.
    """
    import csv
    import io
    import random
    from .evaluate import summarize
    from .dataset import load_dataset, unpack
    data = once('benchmark_dataset', load_dataset)
    tasks = once('benchmark_tasks', lambda: {q.id: (g, q, row['seed']) for row in data
                 for g, queries in [unpack(row)] for q in queries if row['split'] in ('test', 'ood')})
    require(isinstance(text, str), '缺少逐题CSV；不能仅展示汇总数字。')
    rows = list(csv.DictReader(io.StringIO(text)))
    methods = ('bfs', 'astar_admissible', 'astar_fixed', 'astar_ga', 'astar_ml', 'mcts', 'hybrid', 'hybrid_no_expert')
    expected = {(q, b, m) for q in tasks for b in (4, 8, 16) for m in methods}
    require(len(rows) == len(expected) == manifest['runs'], '原基准CSV行数与实验方案不一致。')
    for row in rows:
        for key in ('seed', 'budget', 'hops_requested', 'expanded', 'total_work', 'successor_checks',
                    'heuristic_evaluations', 'heuristic_probe_states', 'proof_pruned', 'proof_checks',
                    'adjacency_edge_scans', 'heuristic_edge_checks', 'total_edge_checks',
                    'successor_cache_hits', 'successor_cache_entries', 'expert_rejections'):
            row[key] = int(row[key]); require(row[key] >= 0, '原基准CSV有负数计数。')
        for key in ('valid', 'found', 'answerable'):
            require(row[key] in ('True', 'False'), '原基准CSV布尔字段不合法。')
            row[key] = row[key] == 'True'
        for key in ('redundancy_ratio', 'redundant_path_ratio', 'elapsed_ms'):
            row[key] = float(row[key]); require(finite(row[key]) and row[key] >= 0, '原基准CSV非有限或负数成本。')
        row['hops'] = int(row['hops']) if row['hops'] else None
        row['peak_frontier'] = int(row['peak_frontier']) if row['peak_frontier'] else None
        graph, query, seed = tasks[row['query_id']]
        require(row['seed'] == seed and row['graph_id'] == graph.id and row['split'] == graph.split
                and row['hops_requested'] == query.hops and row['answerable'] is True
                and (not row['valid'] or row['found']) and row['status'] in ('found', 'exhausted', 'budget_exhausted')
                and row['found'] == (row['status'] == 'found') and row['expanded'] <= row['budget']
                and (row['hops'] == query.hops if row['found'] else row['hops'] is None)
                and row['total_work'] >= row['expanded'] and row['adjacency_edge_scans'] >= row['total_edge_checks']
                and row['total_edge_checks'] == row['successor_checks'] + row['heuristic_edge_checks'],
                '原基准CSV问题、状态或成本关系不一致。')
    require({(r['query_id'], r['budget'], r['method']) for r in rows} == expected, '原基准CSV组合缺失或重复。')
    require(same(summary, summarize(rows)), '原基准展示汇总与逐题CSV不一致。')
    # Same graph-cluster resampling as the frozen experiment, with sufficient
    # integer (sum,count) statistics. Identical results without expanding each
    # cluster into duplicate Python lists 18,000 times on every request.
    computed = []
    for split in ('test', 'ood'):
        for budget in (4, 8, 16):
            base = {r['query_id']: r for r in rows if r['split'] == split and r['budget'] == budget and r['method'] == 'astar_fixed'}
            for method in ('astar_ga', 'hybrid', 'mcts'):
                selected = [r for r in rows if r['split'] == split and r['budget'] == budget and r['method'] == method]
                groups = {}
                for row in selected:
                    groups.setdefault(row['graph_id'], []).append(int(row['valid']) - int(base[row['query_id']]['valid']))
                clusters = [(sum(d), len(d)) for d in groups.values()]
                rng = random.Random(123); samples = []
                for _ in range(1000):
                    draw = rng.choices(clusters, k=len(clusters))
                    samples.append(sum(a for a, b in draw) / sum(b for a, b in draw))
                samples.sort()
                computed.append({'split': split, 'budget': budget, 'method': method, 'baseline': 'astar_fixed',
                                 'success_delta': sum(a for a, b in clusters) / sum(b for a, b in clusters),
                                 'graph_bootstrap_low': samples[25], 'graph_bootstrap_high': samples[974],
                                 'graph_clusters': len(clusters)})
    require(same(paired, computed), '原基准配对区间与逐题CSV图级重采样不一致。')
    index = {(r['query_id'], r['budget'], r['method']): r for r in rows}
    require(isinstance(failures, list), '失败案例记录不完整。')
    seen = set()
    for failure in failures:
        key = (failure['query']['id'], failure['result']['budget'], failure['method'])
        require(key not in seen and key in index, '失败案例重复或未对应逐题记录。')
        seen.add(key);row = index[key];graph, query, _ = tasks[key[0]]
        require(not row['valid'] and failure['graph_id'] == graph.id and digest(failure['query']) == digest(query.__dict__)
                and failure['status'] == row['status'] and all(same(failure['result'].get(k), row[k]) for k in RESULT_FIELDS),
                '失败案例与当前问题或逐题CSV记录不一致。')
    return True
