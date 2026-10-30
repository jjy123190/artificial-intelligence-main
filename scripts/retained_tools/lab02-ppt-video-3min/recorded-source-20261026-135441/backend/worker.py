"""隔离进程中的网页推理适配。只接收数据，不接收任意文件路径或代码。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from core.kb.domain import load_from_equity_kg, demo_case
from core.kb.graph_io import case_from_dict, case_to_dict
from core.kb.rules import build_rules, PRED_CN
from core.kb.rules_dsl import guard_vars
from experiments.lab02.hybrid.pipeline import prepare_case, assess_company
from experiments.lab02.hybrid.model import HybridModel
from experiments.lab02.deterministic.backward import BackwardEngine
from experiments.lab02.deterministic.forward import InferenceLimitError

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = {"graph": ROOT / "outputs" / "lab02" / "end_to_end" / "models.json",
               "indicator": ROOT / "outputs" / "lab02" / "experiments" / "models.json"}


def model_catalog() -> list[dict]:
    items = [{"id": "manual", "name": "教学预设", "detail": "手工CPT · 固定融合权重"}]
    for family, path in MODEL_FILES.items():
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("models_by_seed"), dict):
                continue
            for seed, bundle in data.get("models_by_seed", {}).items():
                if not isinstance(bundle, dict) or not str(seed).isdigit() or len(str(seed)) > 10:
                    continue
                for variant, key in (("selected", "model"), ("fixed", "fixed_model"), ("adaptive", "adaptive_model")):
                    if key in bundle:
                        try:
                            HybridModel.from_dict(bundle[key])
                        except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
                            continue
                        items.append({"id": f"{family}:{seed}:{variant}",
                            "name": f"{'图级' if family == 'graph' else '指标'}训练 · {seed} · { {'selected': '验证权重', 'fixed': '固定权重', 'adaptive': '自动模糊拟合'}[variant] }",
                            "detail": "模型来自本地实测，不能外推为真实监管准确率"})
        except (ValueError, OSError, TypeError):
            continue
    return items


def load_model(source: str) -> HybridModel:
    if source == "manual":
        return HybridModel()
    available = {item["id"] for item in model_catalog()}
    if source not in available:
        raise ValueError("该模型预设不可用，请改用教学预设或先生成相应实验结果")
    family, seed, variant = source.split(":")
    bundle = json.loads(MODEL_FILES[family].read_text(encoding="utf-8-sig"))["models_by_seed"][seed]
    return HybridModel.from_dict(bundle[{"fixed": "fixed_model", "adaptive": "adaptive_model", "selected": "model"}[variant]])


def preset_case(case_id: str) -> dict:
    if case_id not in ("base", "extended"):
        raise ValueError("未知案例")
    case = load_from_equity_kg() if case_id == "base" else demo_case()
    case.source = "第一次课堂案例的本地数据快照" + ("，附加虚构扩展关系" if case_id == "extended" else "")
    return case_to_dict(case)


def fact_id(fact) -> str:
    normalized = [int(v) if isinstance(v, float) and v.is_integer() else v for v in fact]
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:20]


def pretty_fact(fact, names: dict) -> str:
    def name(value):
        return names.get(value, str(value)) if isinstance(value, str) else str(value)
    p, a = fact[0], fact[1:]
    if p == "Holds": return f"{name(a[0])} 持有 {name(a[1])} {float(a[2]) * 100:g}%"
    if p == "Guarantee": return f"{name(a[0])} 为 {name(a[1])} 担保 {float(a[2]):g} 万元"
    if p == "Trade": return f"{name(a[0])} 向 {name(a[1])} 交易 {float(a[2]):g} 万元"
    if p in ("Controls", "DirectControls"): return f"{name(a[0])} 控制 {name(a[1])}"
    if p == "ActualController": return f"{name(a[1])} 是 {name(a[0])} 的实控人候选"
    if p == "BeneficialOwner": return f"{name(a[1])} 是 {name(a[0])} 的受益人候选"
    if p == "Alert": return f"{name(a[0])} · {a[1]}"
    if p == "RiskLevel": return f"{name(a[0])} 的复核优先级为 {a[1]}"
    if p == "RelatedTrade": return f"{name(a[0])} 与 {name(a[1])} 存在关联交易线索"
    return f"{PRED_CN.get(p, p)}（{'，'.join(name(v) for v in a)}）"


def edges_for_case(case: dict) -> list[dict]:
    edges = []
    families = (("holdings", "holding", "holder", "target", "Holds"),
                ("guarantees", "guarantee", "guarantor", "beneficiary", "Guarantee"),
                ("trades", "trade", "seller", "buyer", "Trade"),
                ("kinships", "kinship", "a", "b", "Kin"))
    for field, kind, src, dst, pred in families:
        for i, relation in enumerate(case[field]):
            value = relation["ratio"] if kind == "holding" else relation["relation"] if kind == "kinship" else relation["amount"]
            fact = (pred, relation[src], relation[dst], value)
            label = f"{value * 100:g}%" if kind == "holding" else str(value) if kind == "kinship" else f"{value:g}万"
            edges.append({"id": f"{kind}-{i}", "source": relation[src], "target": relation[dst],
                          "kind": kind, "label": label, "fact_id": fact_id(fact), "index": i})
    return edges


def analyze_payload(payload: dict) -> dict:
    case = case_from_dict(payload["case"])
    context = prepare_case(case, completeness=payload["completeness"])
    model = load_model(payload["model_source"])
    names = {entity.eid: entity.name for entity in case.entities}
    descriptions = {r.name: r.desc for r in context.engine.rules}
    sources = {fact_id(f.as_tuple()): f.source for f in context.facts}
    provenance = {}
    for fact in sorted(context.engine.wm.facts, key=str):
        fid = fact_id(fact)
        why = context.engine.why.get(fact)
        provenance[fid] = {"id": fid, "fact": fact, "label": pretty_fact(fact, names),
            "rule": why[0] if why else None, "rule_description": descriptions.get(why[0], "") if why else "",
            "premises": [fact_id(p) for p in why[1]] if why else [],
            "source": "规则推导" if why else sources.get(fid, "输入记录")}
    decisions = {}
    for company in case.companies:
        d = assess_company(context, company.eid, model, method=payload["method"],
                           threshold=payload["threshold"], include_explanation=False)
        facts = context.established_for(company.eid)
        claim_ids = [fact_id(f) for f in facts if f[0] in ("Alert", "ActualController", "RiskLevel", "BeneficialOwner")]
        order = {"Alert": 0, "ActualController": 1, "RiskLevel": 2, "BeneficialOwner": 3}
        claim_ids.sort(key=lambda key: (order[provenance[key]["fact"][0]], provenance[key]["label"]))
        unlocked = assess_company(context, company.eid, model, method="hybrid_unlocked", threshold=payload["threshold"])
        locked = assess_company(context, company.eid, model, method="hybrid_locked", threshold=payload["threshold"])
        components = model.components(context.observations[company.eid])
        for key in ("provided_facts", "locked_facts"):
            d.pop(key, None)
        d["claim_ids"] = claim_ids
        d["controllers"] = [f[2] for f in facts if f[0] == "ActualController"]
        d["components"] = components
        d["hybrid_comparison"] = {"unlocked": {k: unlocked[k] for k in ("score", "review_required", "review_priority", "risk_class")},
                                  "locked": {k: locked[k] for k in ("score", "review_required", "review_priority", "risk_class")}}
        decisions[company.eid] = d
    steps = [{"cycle": step.cycle, "rule": step.rule, "fact_ids": [fact_id(f) for f in step.facts],
              "description": descriptions.get(step.rule, "")} for step in context.engine.steps]
    derived_edges = [{"id": f"derived-{i}", "source": f[1], "target": f[2], "kind": "control",
                      "label": "控制", "fact_id": fact_id(f), "index": i}
                     for i, f in enumerate(context.engine.query("Controls", "?x", "?y"))]
    data = case_to_dict(case)
    return {"case": data, "completeness": context.completeness,
            "parameters": {k: payload[k] for k in ("method", "threshold", "model_source")},
            "entities": data["entities"], "edges": edges_for_case(data), "derived_edges": derived_edges,
            "decisions": decisions, "provenance": provenance, "steps": steps,
            "stats": {**context.engine.stats(), "preparation_ms": context.preparation_ms,
                      "entities": len(case.entities), "companies": len(case.companies),
                      "persons": len(case.persons), "relations": sum(len(data[k]) for k in ("holdings", "guarantees", "trades", "kinships"))},
            "model": model.as_dict(), "default_company": "C3" if "C3" in decisions else next(iter(decisions))}


def query_payload(payload: dict) -> dict:
    context = prepare_case(case_from_dict(payload["case"]), completeness=payload["completeness"])
    goal = (payload["predicate"], *payload["arguments"])
    if any(a.startswith("?") for a in payload["arguments"]):
        raise ValueError("网页查询请给出具体实体或预警名称，批量变量查询可使用命令行")
    engine = BackwardEngine(context.engine.rules, context.facts.to_tuples(), max_rounds=200, max_tables=12000)
    proof = engine.solve(goal)
    proved = proof is not None
    if proved != context.engine.holds(goal):
        raise RuntimeError("前后向结论不一致")
    return {"goal": goal, "proved": proved, "equivalent": True,
            "lines": proof.lines({r.name: r.desc for r in context.engine.rules}) if proof else [],
            "stats": engine.stats(), "message": "已找到完整证明，正反向结论一致" if proved else "当前知识库无法证明，不等于现实中为假"}


def worker_entry(operation: str, payload: dict, pipe):
    def send(encoded):
        try:
            pipe.send_bytes(encoded)
        except (BrokenPipeError, EOFError, OSError):
            # The parent may have closed the channel after a timeout/cancellation.
            pass

    try:
        if operation == "lab01":
            from experiments.lab01.service import run_task
            value = run_task(payload)
        elif operation == "lab02":
            from experiments.lab02.service import run_task
            value = run_task(payload)
        elif operation == "governance":
            from core.kb.governance import governance_snapshot
            value = governance_snapshot(payload)
        elif operation == "analyze":
            value = analyze_payload(payload)
        elif operation == "query":
            value = query_payload(payload)
        else:
            raise ValueError("未知工作进程操作")
        encoded = json.dumps({"ok": True, "data": value}, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(encoded) > 6 * 1024 * 1024:
            raise ValueError("推理结果过大，请精简案例或使用命令行分析")
        send(encoded)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        send(json.dumps({"ok": False, "code": "invalid_case", "message": str(exc)[:600]}, ensure_ascii=False).encode("utf-8"))
    except InferenceLimitError:
        send(json.dumps({"ok": False, "code": "inference_limit", "message": "推理达到资源预算，结论尚未完整。请精简案例；不能将此状态当作无法证明。"}, ensure_ascii=False).encode("utf-8"))
    except Exception:
        send(json.dumps({"ok": False, "code": "inference_failed", "message": "推理未完成，请检查案例或降低复杂度"}, ensure_ascii=False).encode("utf-8"))
    finally:
        pipe.close()
