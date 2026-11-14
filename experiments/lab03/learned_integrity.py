"""Validate learned search artifacts against their actual train/validation graphs.

These checks do not optimize parameters, run an LLM, write results, or use test
answers to choose a model. Reconstructing training labels and replaying saved GA
champions verifies existing evidence; it is not part of search or its budget.
Successful checks are cached by full content in the bounded runtime verdict cache.
"""
from __future__ import annotations

import math

from .dataset import unpack
from .domain import FEATURE_NAMES, State, enumerate_paths, features
from .learning import probability
from .runtime_integrity import checked_shape, count, finite, require
from .search import search


GA_FITNESS = 'validation valid answer rate - 0.12 * mean(expansions/budget)'


def selected_records(records, seed, split):
    selected = [r for r in records if r['seed'] == seed and r['split'] == split]
    require(bool(selected), '学习产物没有对应的数据划分。')
    ids = [r['graph']['id'] for r in selected]
    require(all(isinstance(g, str) for g in ids) and len(ids) == len(set(ids)),
            '学习产物来源图编号重复或不合法。')
    require(all(r['graph']['split'] == split for r in selected), '图与数据集的划分不一致。')
    return selected


def graph_ids_match(actual, records):
    expected = sorted(r['graph']['id'] for r in records)
    return (isinstance(actual, list) and all(isinstance(g, str) for g in actual)
            and sorted(actual) == expected)


def ga_weights(value):
    return (isinstance(value, list) and len(value) == 6
            and all(finite(x) and 0 <= x <= 5 for x in value) and value[5] == 0)


@checked_shape
def model_schema(model, seed, records):
    require(isinstance(model, dict) and type(seed) is int
            and type(model.get('seed')) is int and model['seed'] == seed
            and model.get('kind') == 'logistic_regression', '路径分类器类型或种子不一致。')
    require(isinstance(model.get('coef'), list) and len(model['coef']) == 6
            and all(finite(x) for x in model['coef']) and finite(model.get('intercept'))
            and model.get('features') == FEATURE_NAMES, '路径分类器参数或特征契约不一致。')
    train = selected_records(records, seed, 'train')
    require(graph_ids_match(model.get('train_graph_ids'), train),
            '路径分类器必须来自当前种子的完整训练图，不能仅凭编号前缀识别。')
    require(all(count(model.get(k)) and model[k] > 0 for k in ('samples', 'positive', 'negative'))
            and model['samples'] == model['positive'] + model['negative']
            and finite(model.get('train_brier')) and 0 <= model['train_brier'] <= 1,
            '路径分类器必须包含完整的成功/失败样本及有效训练统计。')


@checked_shape
def model_training_evidence(model, seed, records):
    """Reconstruct only train-prefix labels and check saved counts and predictions."""
    model_schema(model, seed, records)
    errors = []
    positives = 0
    for record in selected_records(records, seed, 'train'):
        graph, queries = unpack(record)
        for query in queries:
            solutions = enumerate_paths(graph, query)
            positive = {s.edges[:i] for s in solutions for i in range(s.depth + 1)}
            todo = [State((query.start,))]
            while todo:
                state = todo.pop()
                target = int(state.edges in positive)
                predicted = probability(model, features(graph, state, query))
                errors.append((predicted - target) ** 2)
                positives += target
                todo.extend(graph.apply(state, e) for e in graph.successors(state, query, True))
    require(model['samples'] == len(errors) and model['positive'] == positives
            and model['negative'] == len(errors) - positives,
            '路径分类器样本计数与训练图成功/失败前缀不一致。')
    require(math.isclose(model['train_brier'], math.fsum(errors) / len(errors),
                         rel_tol=1e-10, abs_tol=1e-12),
            '路径分类器参数与保存的训练误差不一致，不能使用损坏的学习参数。')


@checked_shape
def ga_schema(record, seed, records):
    require(isinstance(record, dict) and type(seed) is int
            and type(record.get('seed')) is int and record['seed'] == seed,
            'GA种子不一致。')
    require(ga_weights(record.get('weights')) and record.get('fitness') == GA_FITNESS,
            'GA权重或适应度契约不一致。')
    require(type(record.get('population')) is int and record['population'] >= 3
            and count(record.get('generations'))
            and type(record.get('budget')) is int and 1 <= record['budget'] <= 256,
            'GA种群、代数或预算不合法。')
    validation = selected_records(records, seed, 'validation')
    require(graph_ids_match(record.get('validation_graph_ids'), validation)
            and type(record.get('validation_queries')) is int
            and record['validation_queries'] == sum(len(r['queries']) for r in validation),
            'GA验证图及问题数量与当前数据集不一致。')
    initial = record.get('initial_weights')
    require(isinstance(initial, list) and len(initial) < record['population']
            and all(ga_weights(w) for w in initial), 'GA初始候选不完整。')
    maximum = record['population'] + (record['population'] - 2) * record['generations']
    require(count(record.get('unique_candidates'), maximum) and record['unique_candidates'] > 0,
            'GA候选计数与种群/精英保留方案不一致。')
    history = record.get('history')
    require(isinstance(history, list) and len(history) == record['generations'] + 1,
            'GA逐代历史缺失，不能将局部记录当作完整优化。')
    previous = -math.inf
    for generation, row in enumerate(history):
        require(isinstance(row, dict) and type(row.get('generation')) is int
                and row['generation'] == generation and ga_weights(row.get('weights')),
                'GA代号必须连续唯一，并记录该代最优权重。')
        require(finite(row.get('success_rate')) and 0 <= row['success_rate'] <= 1
                and finite(row.get('expanded_fraction')) and 0 <= row['expanded_fraction'] <= 1
                and finite(row.get('fitness')),
                'GA逐代测量值不合法。')
        require(math.isclose(row['fitness'], row['success_rate'] - .12 * row['expanded_fraction'],
                             rel_tol=1e-10, abs_tol=1e-12)
                and row['fitness'] >= previous - 1e-12,
                'GA适应度与成功率/扩展数不一致，或违反精英保留。')
        previous = row['fitness']
    require(record['weights'] == history[-1]['weights'], 'GA部署权重与最后一代最优候选不一致。')


@checked_shape
def ga_validation_evidence(record, seed, records):
    """Replay only stored champions on validation, never evolve or select on test."""
    ga_schema(record, seed, records)
    tasks = [(graph, query) for row in selected_records(records, seed, 'validation')
             for graph, queries in [unpack(row)] for query in queries]
    measured = {}
    for generation in record['history']:
        key = tuple(generation['weights'])
        if key not in measured:
            results = [search(g, q, 'astar_ga', record['budget'], weights=list(key),
                              seed=seed, trace=False) for g, q in tasks]
            measured[key] = (sum(r['valid'] for r in results) / len(results),
                             sum(r['expanded'] for r in results) / (len(results) * record['budget']))
        success, work = measured[key]
        require(math.isclose(generation['success_rate'], success, rel_tol=1e-10, abs_tol=1e-12)
                and math.isclose(generation['expanded_fraction'], work, rel_tol=1e-10, abs_tol=1e-12),
                'GA逐代测量无法在来源验证图重放，不能展示为实测优化。')


@checked_shape
def learning_metrics(rows, models, records):
    """Test metrics are verification-only and never enter training or GA fitness."""
    from .learning import classifier_evaluation
    seeds = (7, 19, 43)
    require(isinstance(rows, list) and len(rows) == len(seeds)
            and all(isinstance(r, dict) and type(r.get('seed')) is int for r in rows)
            and {r['seed'] for r in rows} == set(seeds), '分类器测试记录缺失或种子重复。')
    require(all(str(seed) in models for seed in seeds), '分类器测试结果缺少对应的有效学习模型。')
    for row in rows:
        require(row.get('split') == 'test' and count(row.get('samples')) and row['samples'] > 0
                and all(finite(row.get(k)) and 0 <= row[k] <= 1
                        for k in ('precision', 'recall', 'f1', 'brier')),
                '分类器测试指标类型或范围不正确；不能把未测量/损坏字段显示为零。')
        expected = classifier_evaluation(records, models[str(row['seed'])], split='test')
        require(row['samples'] == expected['samples']
                and all(math.isclose(row[k], expected[k], rel_tol=1e-10, abs_tol=1e-12)
                        for k in ('precision', 'recall', 'f1', 'brier')),
                '分类器测试指标与当前模型的独立测试路径前缀不一致。')
