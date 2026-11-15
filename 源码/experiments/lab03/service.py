"""Bounded course adapter: validate saved artifacts, never silently relabel a fallback."""
from __future__ import annotations

import json
import math
import os

from .dataset import ROOT, load_dataset, unpack, shared_graph
from .domain import State, digest, expert_legal, rejection_reason, path_key
from .artifacts import provenance as _artifact_provenance, source_hashes as _artifact_source_hashes
from .runtime_integrity import (request_scoped, once, audit_index, original_audit,
                                original_llm_summary, confirmation_rows, fresh_llm, benchmark_tables, same, verified_inputs)
from .llm import SCORE_PROTOCOL, MODEL_ID, REVISION, WEIGHTS_SHA256, candidate_actions, path_explanation
from .search import search, METHODS, HYBRID_ML_WEIGHT

OUT = ROOT / "outputs/lab03"
GA_METHODS = {"astar_ga", "hybrid", "hybrid_no_expert", "hybrid_llm"}
ML_METHODS = {"astar_ml", "hybrid", "hybrid_no_expert", "hybrid_llm"}
REBUILD = "请运行 python main.py lab03 benchmark 重新生成实验数据。"


def _nonfinite_json(value):
    raise ValueError('实验记录不允许NaN或Infinity')


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('实验记录包含重复JSON字段：' + key)
        result[key] = value
    return result


def read_json(path):
    """One strict JSON contract for runtime requests and code-only setup checks."""
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=_nonfinite_json,
                      object_pairs_hook=_unique_json_pairs)


def read(name, default=None):
    def load():
        p = OUT / name
        if not p.exists():
            return default
        try:
            return read_json(p)
        except (OSError, ValueError) as exc:
            raise ValueError(f"实验数据 {name} 无法读取或已损坏；{REBUILD}") from exc
    return once(('json', name), load)


def source_hashes():
    return once('source_hashes', _artifact_source_hashes)


def provenance():
    return once('provenance', _artifact_provenance)


def current_artifact(record):
    return isinstance(record, dict) and record.get('provenance') == provenance()


def current_manifest(record):
    return (isinstance(record, dict) and record.get('dataset_sha256') == provenance()['dataset_sha256']
            and isinstance(record.get('source_sha256'), dict)
            and all(record['source_sha256'].get(p) == h for p, h in source_hashes().items()))


def _readout():
    from .calibration import load_readout
    return once('readout', load_readout)


def _readout_sha():
    return once('readout_sha256', lambda: digest(_readout()))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _dataset():
    return once('learned_dataset', load_dataset)


def _dataset_sha():
    # A canonical content key avoids marshal reference-aliasing cache misses when
    # unpacking the same JSON into dataclasses changes its in-memory references.
    return once('learned_dataset_sha256', lambda: digest(_dataset()))


@request_scoped
def valid_model(model, seed):
    from .learned_integrity import model_schema, model_training_evidence
    if not current_artifact(model):
        return False
    try:
        records = _dataset()
        model_schema(model, seed, records)
        verified_inputs('path_model_training', [digest(model), seed, _dataset_sha()],
                        lambda: model_training_evidence(model, seed, records))
        return True
    except (ValueError, OSError):
        return False


@request_scoped
def valid_ga(record, seed):
    from .learned_integrity import ga_schema, ga_validation_evidence
    if not current_artifact(record):
        return False
    try:
        records = _dataset()
        ga_schema(record, seed, records)
        verified_inputs('ga_validation_champions', [digest(record), seed, _dataset_sha()],
                        lambda: ga_validation_evidence(record, seed, records))
        return True
    except (ValueError, OSError):
        return False


@request_scoped
def validate_replay(record, graph, query):
    if not isinstance(record, dict) or not record:
        raise ValueError("该问题没有实测LLM评分。请选择标注“LLM实测”的问题，或使用其他搜索方法。")
    if record.get("graph_sha256") != digest(graph.to_dict()) or record.get("query_sha256") != digest(query.__dict__):
        raise ValueError("LLM评分记录与当前图或问题不一致，请重新生成评分。")
    if record.get('protocol') != SCORE_PROTOCOL:
        raise ValueError('LLM评分协议已更新，请重新运行 python main.py lab03 llm。')
    if record.get('source_sha256') != source_hashes():
        raise ValueError('LLM评分记录的源码指纹已过期，请重新运行 python main.py lab03 llm。')
    readout=_readout()
    if record.get('readout_sha256')!=_readout_sha():
        raise ValueError('LLM评分与有监督证据分类头不一致，请重新生成评分')
    preparation_scans=[]
    candidates = candidate_actions(graph, query, account=preparation_scans.append)
    expected_keys = {path_key(state.edges + (edge.id,)) for state, edge in candidates}
    scores = record.get("scores")
    if (not isinstance(scores, dict) or (expected_keys and not scores)
            or set(scores) != expected_keys
            or any(not finite(v) or not 0 <= v <= 1 for v in scores.values())
            or type(record.get("scored_edges")) is not int or record["scored_edges"] != len(scores)
            or not finite(record.get("llm_ms")) or record["llm_ms"] < 0
            or type(record.get("preparation_edge_scans")) is not int
            or type(record.get("preparation_states")) is not int
            or record.get("preparation_edge_scans") != sum(preparation_scans)
            or record.get("preparation_states") != len(preparation_scans)
            or record.get("model") != MODEL_ID or record.get('revision') != REVISION
            or record.get('weights_sha256') != WEIGHTS_SHA256):
        raise ValueError("LLM评分记录不完整或含无效数值，请重新生成评分。")
    indexed=once('audit_index', lambda: audit_index(read('llm_audit.json')))
    original=once(('checked_audit', query.id, record['graph_sha256'], record['query_sha256']),
                  lambda: original_audit(indexed.get(query.id), graph, query, readout, _readout_sha()))
    if any(not same(record.get(k), v) for k, v in original.items()):
        raise ValueError('LLM缓存评分或推断成本与原始推断记录不一致；不能把修改后的数字当作实测重放。')
    return scores


def tasks():
    return once('display_tasks', _tasks)


def _tasks():
    entries = {}
    graph, queries = shared_graph()
    for q in queries:
        entries[q.id] = (graph, q, 7)
    data = _dataset()
    for seed in [7, 19, 43]:
        record = next(r for r in data if r["split"] == "test" and r["seed"] == seed)
        graph, queries = unpack(record)
        for q in queries[:3]:
            entries[q.id] = (graph, q, seed)
    return entries


def confirmation_summary(record):
    """Small, checked display projection; never send thousands of raw runs to the UI."""
    if not isinstance(record, dict) or record.get('source_sha256') != source_hashes():
        raise ValueError('新图确认记录与当前源码不一致；请重跑实验三llm命令。')
    for key, count in [('question_count', 216), ('paired_runs', 5184),
                       ('negative_questions', 216), ('negative_pairs', 864)]:
        if type(record.get(key)) is not int or record[key] != count:
            raise ValueError('新图确认记录不完整；不能将缺失结果显示为零。')
    splits = {'reliability_confirmation', 'reliability_confirmation_shift'}
    for key, budgets, methods in [('summary', (4, 8, 16), set(METHODS) - {'hybrid_llm'}),
                                  ('negative_summary', (8, 64), {'astar_fixed', 'hybrid'})]:
        rows = record.get(key)
        if not isinstance(rows, list) or len(rows) != len(splits) * len(budgets) * len(methods):
            raise ValueError('新图确认统计矩阵不完整。')
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('新图确认统计行格式错误。')
            split, budget, method = row.get('split'), row.get('budget'), row.get('method')
            if (not isinstance(split, str) or split not in splits or type(budget) is not int
                    or budget not in budgets or not isinstance(method, str) or method not in methods
                    or type(row.get('n')) is not int or row['n'] != 108):
                raise ValueError('新图确认分组与冻结方案不一致。')
            identity = (split, budget, method)
            if identity in seen:
                raise ValueError('新图确认统计分组重复。')
            seen.add(identity)
            for side in ('before', 'after'):
                stats = row.get(side)
                if not isinstance(stats, dict) or any(not finite(stats.get(k)) or stats[k] < 0
                                                     for k in ('mean_expanded', 'mean_edge_checks')):
                    raise ValueError('新图确认统计包含无效成本。')
                counts = ('correct_exhausted', 'false_found') if key == 'negative_summary' else ('invalid_answers',)
                if any(type(stats.get(k)) is not int or not 0 <= stats[k] <= row['n'] for k in counts):
                    raise ValueError('新图确认统计包含无效计数。')
                if key == 'summary' and (not finite(stats.get('success_rate')) or not 0 <= stats['success_rate'] <= 1):
                    raise ValueError('新图确认答案率无效。')
    verified_inputs('confirmation', [source_hashes(), record], lambda: confirmation_rows(record))
    return {k: record[k] for k in ('question_count', 'paired_runs', 'negative_questions', 'negative_pairs',
                                   'summary', 'negative_summary')}


def llm_confirmation_summary(record, readout_sha):
    """Display only complete prediction-to-search-to-metric evidence."""
    if (not isinstance(record, dict) or record.get('source_sha256') != source_hashes()
            or not readout_sha or record.get('readout_sha256') != readout_sha
            or type(record.get('questions')) is not int or record['questions'] != 18
            or type(record.get('prefixes')) is not int or record['prefixes'] != 144):
        raise ValueError('新LLM确认记录与当前模型评估头不一致；请重跑实验三llm命令。')
    models, ga = read('models.json', {}), read('ga_history.json', {})
    if (not isinstance(models, dict) or not isinstance(ga, dict)
            or any(not valid_model(models.get(str(seed)), seed) or not valid_ga(ga.get(str(seed)), seed)
                   for seed in (7, 19, 43))):
        raise ValueError('新LLM确认依赖的模型或GA记录缺失；相关统计暂不展示。')
    training = read('llm_readout_training.json')
    verified_inputs('fresh_llm', [source_hashes(), record, readout_sha, models, ga, training],
                    lambda: fresh_llm(record, _readout(), readout_sha, models, ga, training))
    value = {k: record[k] for k in ('questions', 'prefixes', 'raw_metrics', 'calibrated_metrics',
                                    'novel_prompt_prefixes', 'novel_raw_metrics', 'novel_calibrated_metrics',
                                    'numeric_raw', 'numeric_calibrated', 'paired_summary', 'inference_ms')}
    value['numeric_cases'] = len(record['numeric_challenge'])
    return value


@request_scoped
def snapshot():
    warnings = []

    def optional(name, default):
        try:
            value = read(name)
            if value is None:
                raise ValueError(f"缺少 {name}；相关结果暂不可用。")
            if not isinstance(value, type(default)):
                raise ValueError(f"{name} 数据结构不正确；相关结果暂不可用。")
            return value
        except ValueError as exc:
            warnings.append(str(exc))
            return default

    cache = optional("llm_cache.json", {})
    models = optional("models.json", {})
    ga = optional("ga_history.json", {})
    available_ga = {}
    available_models = {}
    for seed in [7, 19, 43]:
        if valid_ga(ga.get(str(seed)), seed):
            available_ga[str(seed)] = ga[str(seed)]
        else:
            warnings.append(f"种子 {seed} 的GA参数不可用；{REBUILD}")
        if valid_model(models.get(str(seed)), seed):
            available_models[str(seed)] = models[str(seed)]
        else:
            warnings.append(f"种子 {seed} 的路径分类器不可用；{REBUILD}")
    queries = []
    for graph, q, seed in tasks().values():
        reason = ""
        try:
            validate_replay(cache.get(q.id), graph, q)
        except ValueError as exc:
            reason = str(exc)
        unavailable = {}
        for method in METHODS:
            if method in GA_METHODS and str(seed) not in available_ga:
                unavailable[method] = "缺少对应种子的GA参数"
            if method in ML_METHODS and str(seed) not in available_models:
                unavailable[method] = "缺少对应种子的路径分类器"
        queries.append({"id": q.id, "text": q.text, "hops": q.hops, "graph_id": graph.id,
                        "start_name": graph.nodes[q.start].name, "relations": list(q.relations),
                        "sector": q.sector, "listed": q.listed,
                        "split": graph.split, "seed": seed, "nodes": len(graph.nodes),
                        "edges": len(graph.edges), "llm_cached": not reason,
                        "llm_reason": reason, "unavailable_methods": unavailable})
    data = {"queries": queries, "methods": [{"id": k, "name": v} for k, v in METHODS.items()],
            "ga": available_ga, "remote_available": bool(os.environ.get("LAB03_LLM_API_KEY") and os.environ.get("LAB03_LLM_MODEL")),
            "notes": ["所有benchmark图与标签为教学构造；shared图来自前两次实验。",
                      "端点由搜索与谓词验证产生，LLM只给路径前缀评分；解释在搜索后生成。",
                      "GA仅在验证图优化；测试/OOD不参与调参。",
                      "LLM价值与合法性输入只有路径事实，不包含专家规则计算出的标签。",
                      "BFS/A*按出队状态计费，MCTS按树扩展计费；total_work仅计扩展+模拟；有界后缀探测、证明剪枝、全部出边扫描与LLM训练/推断另列。"]}
    for key, name, default in [
        ("summary", "summary.json", []), ("learning", "learning_metrics.json", []),
        ("manifest", "manifest.json", {}), ("paired", "paired_effects.json", []),
        ("failures", "failure_cases.json", []), ("llm_summary", "llm_summary.json", {}),
        ("llm_explanations", "llm_explanations.json", []), ("stress", "hallucination_stress.json", {}),
    ]:
        data[key] = optional(name, default)
    if not current_manifest(data['manifest']):
        warnings.append('基准结果与当前源码或数据指纹不一致；旧统计已隐藏。' + REBUILD)
        for key, empty in [('summary', []), ('paired', []), ('failures', []), ('learning', []), ('manifest', {})]:
            data[key] = empty
    if data['manifest']:
        try:
            if len(available_models) != 3 or len(available_ga) != 3:
                raise ValueError('缺少完整有效的模型/GA记录，不能将旧表与当前搜索混用。')
            csv_text = (OUT/'runs.csv').read_text(encoding='utf-8-sig')
            verified_inputs('benchmark', [source_hashes(), csv_text, data['summary'], data['paired'],
                                           data['failures'], data['manifest']],
                            lambda: benchmark_tables(csv_text, data['summary'], data['paired'],
                                                     data['failures'], data['manifest']))
        except (OSError, ValueError) as exc:
            warnings.append('原基准统计核验失败：' + str(exc))
            for key, empty in [('summary', []), ('paired', []), ('failures', []), ('learning', []), ('manifest', {})]:
                data[key] = empty
    if data['manifest']:
        try:
            from .learned_integrity import learning_metrics
            records = _dataset()
            verified_inputs('path_model_test_metrics', [digest(data['learning']), digest(available_models), _dataset_sha()],
                            lambda: learning_metrics(data['learning'], available_models, records))
        except (ValueError, OSError) as exc:
            warnings.append('路径分类器测试记录核验失败：' + str(exc))
            data['learning'] = []
    try:
        readout_sha=_readout_sha()
    except ValueError:
        readout_sha=None
    llm_source = data['llm_summary'].get('source_sha256')
    if (data['llm_summary'].get('protocol') != SCORE_PROTOCOL or llm_source != source_hashes()
            or data['llm_summary'].get('model') != MODEL_ID or data['llm_summary'].get('revision') != REVISION
            or data['llm_summary'].get('weights_sha256') != WEIGHTS_SHA256
            or data['llm_summary'].get('dataset_sha256') != provenance()['dataset_sha256']
            or readout_sha is None or data['llm_summary'].get('readout_sha256')!=readout_sha):
        warnings.append('LLM分析记录与当前协议或源码不一致；请重新运行 python main.py lab03 llm。')
        data['llm_summary'], data['llm_explanations'], data['stress'] = {}, [], {}
    if data['llm_summary']:
        try:
            # All selected replay records must have passed the raw-evidence binding.
            expected = [q for q in queries if q['split'] == 'test']
            if any(not q['llm_cached'] for q in expected):
                raise ValueError('部分LLM缓存未通过原始推断核验；整组统计暂不可用。')
            raw_audit, runs, diagnostics = read('llm_audit.json'), read('llm_runs.json'), read('llm_diagnostics.json')
            verified_inputs('original_llm', [source_hashes(), provenance(), data['llm_summary'], raw_audit,
                            runs, data['llm_explanations'], diagnostics, data['stress'], cache,
                            available_models, available_ga, readout_sha],
                            lambda: original_llm_summary(data['llm_summary'], raw_audit, runs,
                                data['llm_explanations'], diagnostics, data['stress'], tasks(), cache,
                                available_models, available_ga, _readout(), readout_sha, source_hashes(), provenance()))
        except ValueError as exc:
            warnings.append(str(exc))
            data['llm_summary'], data['llm_explanations'], data['stress'] = {}, [], {}
    data['confirmation'], data['llm_confirmation'] = {}, {}
    if current_manifest(data['manifest']):
        for key, name, project in [
            ('confirmation', 'reliability/confirmation.json', confirmation_summary),
            ('llm_confirmation', 'reliability/llm_confirmation.json',
             lambda value: llm_confirmation_summary(value, readout_sha)),
        ]:
            try:
                data[key] = project(optional(name, {}))
            except ValueError as exc:
                warnings.append(str(exc))
    data["warnings"] = list(dict.fromkeys(warnings))
    return data


def explain_result(graph, result):
    if result["valid"]:
        return " → ".join(graph.nodes[n].name for n in result["path"])
    if result["found"]:
        return "搜索返回了候选路径，但证据未通过核验，不能作为有效答案。请查看下方问题证据，或启用专家约束重新搜索。"
    if result["status"] == "budget_exhausted":
        return "扩展预算已用完，仍有候选状态未访问。可以增加预算再试；当前结果不表示图中无解。"
    return "允许的不重复实体路径已全部检查；在当前固定关系序列与专家证据约束下没有答案。该结论不代表开放世界中不存在关系。增加预算不会改变本次结论。"


@request_scoped
def run_task(payload):
    if not isinstance(payload,dict):raise ValueError('实验三请求必须是对象')
    qid = payload.get("query_id", "shared-3")
    if not isinstance(qid,str) or qid not in tasks():
        raise ValueError("未知实验三问题")
    graph, query, seed = tasks()[qid]
    method = payload.get("method", "hybrid")
    budget = payload.get("budget", 16)
    if not isinstance(method,str) or method not in METHODS:
        raise ValueError("未知搜索方法")
    if type(budget) is not int or not 1 <= budget <= 128:
        raise ValueError("网页扩展预算必须是1–128的整数")
    weights, model = None, None
    if method in GA_METHODS:
        ga = read("ga_history.json", {})
        record = ga.get(str(seed)) if isinstance(ga, dict) else None
        if not valid_ga(record, seed):
            raise ValueError(f"种子 {seed} 的GA参数缺失或损坏，不能用固定参数冒充GA；{REBUILD}")
        weights = record["weights"][:]
        weights[5] = HYBRID_ML_WEIGHT
    if method in ML_METHODS:
        models = read("models.json", {})
        model = models.get(str(seed)) if isinstance(models, dict) else None
        if not valid_model(model, seed):
            raise ValueError(f"种子 {seed} 的路径分类器缺失或损坏，不能用中性概率冒充学习模型；{REBUILD}")
    scores = None
    llm_meta = {"mode": "off"}
    if method == "hybrid_llm":
        mode = payload.get("llm_mode", "replay")
        if mode == "replay":
            cache = read("llm_cache.json", {})
            record = cache.get(qid) if isinstance(cache, dict) else None
            scores = validate_replay(record, graph, query)
            llm_meta = {k: record[k] for k in ["model", "scored_edges", "llm_ms", "graph_sha256", "query_sha256", "preparation_edge_scans", "preparation_states"]}
            llm_meta.update({"mode": "replay", "note": "重放真实Qwen评分与训练图拟合的证据分类头；本次未调用LLM。", "scores": scores})
        elif mode == "remote":
            from .llm import score_actions
            if not os.environ.get("LAB03_LLM_API_KEY") or not os.environ.get("LAB03_LLM_MODEL"):
                raise ValueError("远程模型尚未配置，请在服务环境设置LAB03_LLM_API_KEY和LAB03_LLM_MODEL。")
            from .llm import bound_scores
            preparation_scans=[]
            candidates=candidate_actions(graph, query, account=preparation_scans.append)
            records = score_actions(graph, query, candidates, "remote")
            scores = bound_scores(records)
            llm_meta = {"mode": "remote", "records": records, "model": os.environ["LAB03_LLM_MODEL"],
                        "scored_edges": len(records), "preparation_edge_scans":sum(preparation_scans),
                        "preparation_states":len(preparation_scans), "llm_ms": sum(r["inference_ms"] for r in records),
                        "protocol": SCORE_PROTOCOL, "note": "远程只使用原始模型输出，不套用本地Qwen分类头；前两层最多8个真实路径前缀，其余中性评分。后端不同，不能当作本地校准配对结论。"}
        else:
            raise ValueError("不支持的LLM模式")
    result = search(graph, query, method, budget, weights, model, seed, llm_scores=scores)
    evidence = [{**graph.edges[e].__dict__, "eligible": expert_legal(graph.edges[e]),
                 "rejection_reason": rejection_reason(graph.edges[e])} for e in result["edge_ids"]]
    return {"query": query.__dict__, "graph": graph.to_dict(), "result": result, "llm": llm_meta,
            "path_evidence": evidence, "explanation": path_explanation(graph, query, result) if result["valid"] else explain_result(graph, result)}
