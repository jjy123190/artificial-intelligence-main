"""Audit the saved optimization against unchanged questions and a preserved baseline.

Selection evidence is validation-only. Historical test/OOD have been seen before;
this is a same-question regression comparison, not a fresh confirmatory trial.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import zipfile
from .dataset import ROOT
from .artifacts import source_hashes, provenance

OUT = ROOT / 'outputs/lab03'
FOLDER = OUT / 'optimization'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    indexed = {(r['query_id'], int(r['budget']), r['method']): r for r in rows}
    assert len(indexed) == len(rows), ('重复运行记录', str(path))
    return indexed


def audit_optimization():
    protocol = read(FOLDER/'protocol.json')
    assert protocol['selection_split'] == 'validation' and protocol['classifier_fit_split'] == 'train'
    assert protocol['baseline_dataset_sha256'] == provenance()['dataset_sha256'], '基准数据被改动'
    for section in ('validation_artifact_sha256', 'baseline_evidence_sha256'):
        for name, expected in protocol[section].items():
            assert hashlib.sha256((FOLDER/name).read_bytes()).hexdigest() == expected, ('优化证据变动', name)
    stage = OUT/'refinement'
    with zipfile.ZipFile(stage/'stage_one_baseline.zip') as archive:
        for name, expected in protocol['frozen_search_sources_sha256'].items():
            assert hashlib.sha256(archive.read('experiments/lab03/'+name)).hexdigest() == expected, ('历史冻结策略变动', name)
    for name, expected in protocol['validation_script_sha256'].items():
        assert hashlib.sha256((FOLDER/'validation_scripts'/name).read_bytes()).hexdigest() == expected
    selection = protocol['selected']
    sweeps = read(FOLDER/'validation-search-sweep.json')
    candidates = [r for r in sweeps if r['mode'] == selection['features'] and r['ga_budget'] == selection['ga_budget']]
    assert len(candidates) == 1
    from .search import HYBRID_ML_WEIGHT
    assert HYBRID_ML_WEIGHT == selection['hybrid_ml_weight'] > 0
    ga = read(stage/'previous_ga_history.json')
    for seed, weights in candidates[0]['weights'].items():
        assert ga[seed]['budget'] == selection['ga_budget'] and ga[seed]['weights'] == weights
    llm_sweeps = read(FOLDER/'validation-llm-sweep.json')
    assert all(r['query'].startswith('validation-') for variant in llm_sweeps for r in variant['records'])
    picked = next(r for r in llm_sweeps if r['variant'] == selection['legality_prompt'])
    assert picked['balanced_accuracy'] == max(r['balanced_accuracy'] for r in llm_sweeps)
    from .llm import legality_prompt_for, LEGALITY_TOKENS
    assert LEGALITY_TOKENS == ('True', 'False')
    assert all(legality_prompt_for(r['prompt']) == r['variant_prompt'] for r in picked['records'])
    before, after = csv_rows(FOLDER/'baseline_runs.csv'), csv_rows(stage/'previous_runs.csv')
    assert len(before) == len(after) == 3888 and set(before) == set(after), '比较题目、方法或预算发生改变'
    groups = {}
    regressions = []
    for key, old in before.items():
        new = after[key]
        assert all(old[k] == new[k] for k in ('seed', 'split', 'graph_id', 'hops_requested', 'answerable'))
        old_valid, new_valid = old['valid'] == 'True', new['valid'] == 'True'
        # Blind kernels should be identical, including rollout budget and the selected witness status.
        if key[2] in ('bfs', 'mcts', 'astar_admissible'):
            assert all(old[k] == new[k] for k in ('valid', 'found', 'expanded', 'total_work', 'status', 'hops'))
        assert int(new['total_edge_checks']) == int(new['successor_checks']) + int(new['heuristic_edge_checks'])
        group = groups.setdefault((new['split'], key[1], key[2]), [])
        group.append((old_valid, new_valid, int(old['expanded']), int(new['expanded'])))
        if old_valid and not new_valid:
            regressions.append({'query_id': key[0], 'budget': key[1], 'method': key[2], 'split': new['split'],
                                'before_status': old['status'], 'after_status': new['status']})
    old_summary = {(r['split'], r['budget'], r['method']):r for r in protocol['baseline_summary']}
    new_summary = {(r['split'], r['budget'], r['method']):r for r in read(stage/'previous_summary.json')}
    assert set(groups) == set(old_summary) == set(new_summary)
    comparisons = []
    for (split, budget, method), rows in groups.items():
        n = len(rows)
        for summary, valid_index, work_index in [(old_summary,0,2),(new_summary,1,3)]:
            saved = summary[(split,budget,method)]
            assert saved['n'] == n
            assert math.isclose(saved['success_rate'], sum(r[valid_index] for r in rows)/n)
            assert math.isclose(saved['mean_expanded'], sum(r[work_index] for r in rows)/n)
        comparisons.append({'split': split, 'budget': budget, 'method': method, 'n': n,
                            'before_success_rate': sum(a for a, b, x, y in rows)/n,
                            'after_success_rate': sum(b for a, b, x, y in rows)/n,
                            'newly_solved': sum(not a and b for a, b, x, y in rows),
                            'lost_solutions': sum(a and not b for a, b, x, y in rows),
                            'before_mean_expanded': sum(x for a, b, x, y in rows)/n,
                            'after_mean_expanded': sum(y for a, b, x, y in rows)/n})
    baseline_llm, current_llm = protocol['baseline_llm_summary'], read(stage/'previous_llm_summary.json')
    return {'protocol': 'validation-onehop-v1', 'status': 'passed', 'dataset_sha256': provenance()['dataset_sha256'],
            'source_sha256':read(stage/'previous_manifest.json')['source_sha256'], 'same_question_pairs': len(before),
            'selection': selection, 'validation_queries': 81, 'validation_llm_prefixes': picked['n'],
            'comparisons': comparisons, 'regressions': regressions,
            'llm': {'before_legal_metrics': baseline_llm['legal_metrics'], 'after_legal_metrics': current_llm['legal_metrics'],
                    'before_results': baseline_llm['results'], 'after_results': current_llm['results'],
                    'current_paired_diagnostics': current_llm['paired_diagnostics']},
            'limits': protocol['limits'],
            'cost_note': 'Expanded nodes are not total compute: heuristic_edge_checks and total_edge_checks count extra one-hop edge inspections; LLM and training times remain separate.'}


if __name__ == '__main__':
    result = audit_optimization()
    (FOLDER/'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'same_question_pairs': result['same_question_pairs'],
                      'comparison_groups': len(result['comparisons']), 'regressions_retained': len(result['regressions'])}, ensure_ascii=False, indent=2))
