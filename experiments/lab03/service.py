"""Bounded course adapter: validate saved artifacts, never silently relabel a fallback."""
from __future__ import annotations

from functools import lru_cache
import json
import math
import os

from .dataset import ROOT, load_dataset, unpack, shared_graph
from .domain import State, digest, expert_legal, rejection_reason
from .search import search, METHODS

OUT = ROOT / "outputs/lab03"
GA_METHODS = {"astar_ga", "hybrid", "hybrid_no_expert", "hybrid_llm"}
ML_METHODS = {"astar_ml", "hybrid", "hybrid_no_expert", "hybrid_llm"}
REBUILD = "请运行 python main.py lab03 benchmark 重新生成实验数据。"


def read(name, default=None):
    p = OUT / name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"实验数据 {name} 无法读取或已损坏；{REBUILD}") from exc


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def valid_model(model, seed):
    return (isinstance(model, dict) and model.get("seed") == seed
            and isinstance(model.get("coef"), list) and len(model["coef"]) == 6
            and all(finite(x) for x in model["coef"]) and finite(model.get("intercept"))
            and isinstance(model.get("train_graph_ids"), list) and bool(model["train_graph_ids"])
            and all(isinstance(g, str) and g.startswith(f"train-{seed}-") for g in model["train_graph_ids"]))


def valid_ga(record, seed):
    return (isinstance(record, dict) and record.get("seed") == seed
            and isinstance(record.get("weights"), list) and len(record["weights"]) == 6
            and all(finite(x) and x >= 0 for x in record["weights"])
            and isinstance(record.get("history"), list) and bool(record["history"])
            and type(record.get("population")) is int and record["population"] > 0
            and type(record.get("generations")) is int and record["generations"] >= 0
            and type(record.get("validation_queries")) is int and record["validation_queries"] > 0
            and all(isinstance(h, dict) and type(h.get("generation")) is int
                    and finite(h.get("fitness")) and finite(h.get("success_rate"))
                    and 0 <= h["success_rate"] <= 1 and finite(h.get("expanded_fraction"))
                    and 0 <= h["expanded_fraction"] <= 1 for h in record["history"])
            and isinstance(record.get("validation_graph_ids"), list) and bool(record["validation_graph_ids"])
            and all(isinstance(g, str) and g.startswith(f"validation-{seed}-") for g in record["validation_graph_ids"]))


def validate_replay(record, graph, query):
    if not isinstance(record, dict) or not record:
        raise ValueError("该问题没有实测LLM评分。请选择标注“LLM实测”的问题，或使用其他搜索方法。")
    if record.get("graph_sha256") != digest(graph.to_dict()) or record.get("query_sha256") != digest(query.__dict__):
        raise ValueError("LLM评分记录与当前图或问题不一致，请重新生成评分。")
    scores = record.get("scores")
    if (not isinstance(scores, dict) or not scores
            or any(e not in graph.edges or not finite(v) or not 0 <= v <= 1 for e, v in scores.items())
            or type(record.get("scored_edges")) is not int or record["scored_edges"] != len(scores)
            or not finite(record.get("llm_ms")) or record["llm_ms"] < 0
            or not isinstance(record.get("model"), str) or not record["model"]):
        raise ValueError("LLM评分记录不完整或含无效数值，请重新生成评分。")
    return scores


@lru_cache(maxsize=1)
def tasks():
    entries = {}
    graph, queries = shared_graph()
    for q in queries:
        entries[q.id] = (graph, q, 7)
    data = load_dataset()
    for seed in [7, 19, 43]:
        record = next(r for r in data if r["split"] == "test" and r["seed"] == seed)
        graph, queries = unpack(record)
        for q in queries[:3]:
            entries[q.id] = (graph, q, seed)
    return entries


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
    for seed in [7, 19, 43]:
        if valid_ga(ga.get(str(seed)), seed):
            available_ga[str(seed)] = ga[str(seed)]
        else:
            warnings.append(f"种子 {seed} 的GA参数不可用；{REBUILD}")
        if not valid_model(models.get(str(seed)), seed):
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
            if method in ML_METHODS and not valid_model(models.get(str(seed)), seed):
                unavailable[method] = "缺少对应种子的路径分类器"
        queries.append({"id": q.id, "text": q.text, "hops": q.hops, "graph_id": graph.id,
                        "split": graph.split, "seed": seed, "nodes": len(graph.nodes),
                        "edges": len(graph.edges), "llm_cached": not reason,
                        "llm_reason": reason, "unavailable_methods": unavailable})
    data = {"queries": queries, "methods": [{"id": k, "name": v} for k, v in METHODS.items()],
            "ga": available_ga, "remote_available": bool(os.environ.get("LAB03_LLM_API_KEY") and os.environ.get("LAB03_LLM_MODEL")),
            "notes": ["所有benchmark图与标签为教学构造；shared图来自前两次实验。",
                      "端点由搜索与谓词验证产生，LLM只给候选边评分。",
                      "GA仅在验证图优化；测试/OOD不参与调参。",
                      "BFS/A*按出队状态计费，MCTS按树扩展计费；模拟另计total_work。"]}
    for key, name, default in [
        ("summary", "summary.json", []), ("learning", "learning_metrics.json", []),
        ("manifest", "manifest.json", {}), ("paired", "paired_effects.json", []),
        ("failures", "failure_cases.json", []), ("llm_summary", "llm_summary.json", {}),
        ("llm_explanations", "llm_explanations.json", []), ("stress", "hallucination_stress.json", {}),
    ]:
        data[key] = optional(name, default)
    data["warnings"] = list(dict.fromkeys(warnings))
    return data


def explain_result(graph, result):
    if result["valid"]:
        return " → ".join(graph.nodes[n].name for n in result["path"])
    if result["found"]:
        return "搜索返回了候选路径，但证据未通过核验，不能作为有效答案。请查看下方问题证据，或启用专家约束重新搜索。"
    if result["status"] == "budget_exhausted":
        return "扩展预算已用完，仍有候选状态未访问。可以增加预算再试；当前结果不表示图中无解。"
    return "允许的候选路径已全部检查，当前图与关系约束下没有满足条件的答案。增加预算不会改变本次结论。"


def run_task(payload):
    qid = payload.get("query_id", "shared-3")
    if qid not in tasks():
        raise ValueError("未知实验三问题")
    graph, query, seed = tasks()[qid]
    method = payload.get("method", "hybrid")
    budget = payload.get("budget", 16)
    if method not in METHODS:
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
        weights[5] = 1.5
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
            llm_meta = {k: record[k] for k in ["model", "scored_edges", "llm_ms", "graph_sha256", "query_sha256"]}
            llm_meta.update({"mode": "replay", "note": "重放真实小模型评分；本次未调用LLM。", "scores": scores})
        elif mode == "remote":
            from .llm import score_actions
            if not os.environ.get("LAB03_LLM_API_KEY") or not os.environ.get("LAB03_LLM_MODEL"):
                raise ValueError("远程模型尚未配置，请在服务环境设置LAB03_LLM_API_KEY和LAB03_LLM_MODEL。")
            edges = graph.successors(State((query.start,)), query, False)[:6]
            records = score_actions(graph, query, edges, "remote")
            scores = {r["edge_id"]: r["score"] if r.get("legal", True) else .25 * r["score"] for r in records}
            llm_meta = {"mode": "remote", "records": records, "model": os.environ["LAB03_LLM_MODEL"],
                        "scored_edges": len(records), "llm_ms": sum(r["inference_ms"] for r in records),
                        "note": "实时模式仅评估起点的前6条候选边，其余使用中性评分；与离线两层8边实验的协议不同，不作直接性能对比。"}
        else:
            raise ValueError("不支持的LLM模式")
    result = search(graph, query, method, budget, weights, model, seed, llm_scores=scores)
    evidence = [{**graph.edges[e].__dict__, "eligible": expert_legal(graph.edges[e]),
                 "rejection_reason": rejection_reason(graph.edges[e])} for e in result["edge_ids"]]
    return {"query": query.__dict__, "graph": graph.to_dict(), "result": result, "llm": llm_meta,
            "path_evidence": evidence, "explanation": explain_result(graph, result)}
