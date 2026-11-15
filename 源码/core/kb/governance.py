"""日期快照、凭证和身份状态、明确表决权/协议控制的教学治理模型。

验证人是人工登记，不假装对现实身份/凭证作权威核验。证据失效时先过滤输入再
重建最小闭包；记录新旧结论差异，而非在单调推理器中混入未定义的否定规则。
"""
from __future__ import annotations
import copy
import hashlib
from datetime import date
from .graph_io import case_from_dict, case_to_dict
from .rules import build_rules
from .rules_dsl import Rule, Pattern, GuardSpec, guard_vars
from .facts import build_facts, RawFact
from experiments.lab02.deterministic.forward import ForwardEngine
from .equity import economic_equity

RELATIONS = {"holdings": ("holder", "target"), "guarantees": ("guarantor", "beneficiary"),
             "trades": ("seller", "buyer"), "kinships": ("a", "b")}


def day(value):
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError("日期需要YYYY-MM-DD")
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("日期需要YYYY-MM-DD")
    return parsed


def active_window(record, as_of):
    start, end = record.get("valid_from", ""), record.get("valid_to", "")
    if start and end and day(start) >= day(end):
        raise ValueError("有效期结束必须晚于开始；结束日不包含在有效期内")
    start_date = day(start) if start else None
    end_date = day(end) if end else None
    return (start_date is None or start_date <= as_of) and (end_date is None or as_of < end_date)


def relation_key(kind, relation):
    a, b = RELATIONS[kind]
    return f"{kind}:{relation[a]}:{relation[b]}"


def _unique(records, field, label):
    keys = [r[field] for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label}标识重复")


def governance_snapshot(payload: dict) -> dict:
    case = case_from_dict(payload["case"])
    as_of = day(payload["as_of"])
    documents, identities = payload.get("documents", []), payload.get("identities", [])
    metadata, rights = payload.get("relations", []), payload.get("rights", [])
    _unique(documents, "id", "凭证")
    _unique(identities, "entity_id", "身份")
    _unique(metadata, "key", "关系")
    _unique(rights, "id", "权利")
    entity_ids = {e.eid for e in case.entities}
    company_ids = {e.eid for e in case.companies}
    docs = {}
    for raw in documents:
        record = dict(raw)
        digest = hashlib.sha256(record["content"].encode("utf-8")).hexdigest()
        if record.get("sha256") and record["sha256"] != digest:
            raise ValueError("凭证内容摘要不匹配")
        record["sha256"] = digest
        if record["status"] == "verified" and (not record["reviewer"].strip() or not record["content"].strip()):
            raise ValueError("确认凭证必须有核验人及内容；确认不等于真实性认证")
        within_dates = active_window(record, as_of)
        record["usable"] = record["status"] == "verified" and within_dates
        docs[record["id"]] = record
    identities_by_id = {}
    identifiers = {}
    for record in identities:
        if record["entity_id"] not in entity_ids:
            raise ValueError("身份引用未知实体")
        if record["evidence_id"] and record["evidence_id"] not in docs:
            raise ValueError("身份引用未知凭证")
        identifier = record["identifier"].strip().casefold()
        if identifier and identifier in identifiers and identifiers[identifier] != record["entity_id"]:
            raise ValueError("同一核验标识不能绑定两个实体；需先人工处理身份对齐")
        identifiers[identifier] = record["entity_id"]
        verified = (record["status"] == "verified" and bool(record["reviewer"].strip())
                    and bool(identifier) and docs.get(record["evidence_id"], {}).get("usable", False))
        identities_by_id[record["entity_id"]] = {**record, "usable": verified}
    available = {relation_key(kind, r) for kind in RELATIONS for r in payload["case"].get(kind, [])}
    meta = {}
    for record in metadata:
        if record["key"] not in available:
            raise ValueError("关系有效期引用不存在的关系")
        if record["evidence_id"] and record["evidence_id"] not in docs:
            raise ValueError("关系引用未知凭证")
        active_window(record, as_of)
        meta[record["key"]] = record
    filtered = copy.deepcopy(payload["case"])
    excluded, retained = [], []
    for kind, fields in RELATIONS.items():
        filtered[kind] = []
        for relation in payload["case"].get(kind, []):
            key = relation_key(kind, relation)
            record = dict(meta.get(key, {}))
            if kind == "holdings" and not record.get("valid_from") and len(relation.get("since", "")) == 10:
                record["valid_from"] = relation["since"]
            reason = ""
            if not active_window(record, as_of):
                reason = "关系不在有效期"
            elif record.get("evidence_id") and not docs[record["evidence_id"]]["usable"]:
                reason = "凭证未确认、已撤销或过期"
            elif payload.get("strict_evidence", False) and not record.get("evidence_id"):
                reason = "严格凭证模式：关系缺少核验凭证"
            elif payload.get("strict_identity", False) and not all(
                    identities_by_id.get(relation[f], {}).get("usable", False) for f in fields):
                reason = "严格身份模式：关系主体未完成凭证核验"
            if reason:
                excluded.append({"key": key, "reason": reason})
            else:
                filtered[kind].append(relation)
                retained.append({"key": key, "evidence_id": record.get("evidence_id", ""),
                                 "verified": bool(record.get("evidence_id")), "valid_from": record.get("valid_from", ""),
                                 "valid_to": record.get("valid_to", "")})
    effective = case_from_dict(filtered)
    verified_rights, blocked_rights = [], []
    for record in rights:
        if record["party"] not in entity_ids or record["target"] not in company_ids or record["party"] == record["target"]:
            raise ValueError("协议/投票权主体与目标必须存在且不同")
        if record["evidence_id"] not in docs:
            raise ValueError("协议/投票权必须引用登记凭证")
        if record["kind"] not in ("votes", "agreement"):
            raise ValueError("未知权利类型")
        if record["kind"] == "votes" and not 0 <= record["share"] <= 1:
            raise ValueError("投票权份额须在[0,1]")
        if not active_window(record, as_of) or not docs[record["evidence_id"]]["usable"]:
            blocked_rights.append({"id": record["id"], "reason": "权利或凭证未生效/已失效"})
        elif payload.get("strict_identity", False) and not all(identities_by_id.get(e, {}).get("usable", False) for e in (record["party"], record["target"])):
            blocked_rights.append({"id": record["id"], "reason": "权利主体身份未核验"})
        else:
            verified_rights.append(record)
    targets = {r["target"] for r in verified_rights}
    conflicts = []
    accepted = []
    for target in sorted(targets):
        group = [r for r in verified_rights if r["target"] == target]
        votes = [r for r in group if r["kind"] == "votes"]
        agreements = [r for r in group if r["kind"] == "agreement"]
        reason = ""
        if len({r["party"] for r in votes}) != len(votes) or sum(r["share"] for r in votes) > 1 + 1e-9:
            reason = "投票权重复或总量超过100%"
        elif len({r["party"] for r in agreements}) > 1:
            reason = "多个互斥控制协议，需人工裁决"
        elif votes and agreements and any(r["share"] > .5 and r["party"] != agreements[0]["party"] for r in votes):
            reason = "过半投票权与控制协议主体冲突"
        if reason:
            conflicts.append({"target": target, "reason": reason, "right_ids": [r["id"] for r in group]})
        else:
            accepted.extend(group)
    facts = build_facts(effective)
    if any(r["key"].startswith("trades:") for r in excluded):
        # 交易证据失效只意味着未知，不能凭“现在未见交易”新增空壳事实。
        facts.facts = [f for f in facts if f.pred != "Shell"]
    relation_sources = {}
    for kind, fields in RELATIONS.items():
        predicate = {"holdings": "Holds", "guarantees": "Guarantee", "trades": "Trade", "kinships": "Kin"}[kind]
        for relation in filtered[kind]:
            key = relation_key(kind, relation)
            record = meta.get(key, {})
            if record.get("evidence_id"):
                relation_sources[(predicate, relation[fields[0]], relation[fields[1]])] = f"有效关系 {key}；人工凭证 {record['evidence_id']}；快照 {payload['as_of']}"
    facts.facts = [RawFact(f.pred, f.args, relation_sources.get((f.pred, *f.args[:2]), f.source))
                   for f in facts]
    for record in accepted:
        source = f"人工核验凭证 {record['evidence_id']}；权利记录 {record['id']}"
        if record["kind"] == "votes":
            facts.add("VotingShare", record["party"], record["target"], record["share"], record["evidence_id"], source=source)
        else:
            facts.add("ControlAgreement", record["party"], record["target"], record["evidence_id"], source=source)
    rules = build_rules()
    # Explicit voting registers override equity-as-votes assumptions for that target. Conflicts block both.
    for rule in rules:
        if rule.name in ("R01", "R04", "R05", "R05b"):
            target_var = "?y" if rule.name == "R01" else "?c"
            original = rule.guard
            rule.guard = GuardSpec(guard_vars(original) | frozenset([target_var]),
                lambda b, g=original, v=target_var: b[v] not in targets and (g(b) if g else True))
    rules.add(Rule("G01", "核验投票权过半支持控制候选", (Pattern.of("VotingShare", "?p", "?c", "?s", "?e"),),
                   Pattern.of("DirectControls", "?p", "?c"),
                   GuardSpec(frozenset(["?s"]), lambda b: b["?s"] > .5), 10, "凭证控制", "仅教学核验记录，不构成法律认定"))
    rules.add(Rule("G02", "有效且无冲突的控制协议支持控制候选", (Pattern.of("ControlAgreement", "?p", "?c", "?e"),),
                   Pattern.of("DirectControls", "?p", "?c"), priority=10, group="凭证控制"))
    engine = ForwardEngine(rules, facts.to_tuples(), max_cycles=200, max_facts=50000)
    engine.run()
    derived = sorted(engine.why, key=str)
    claims = [f for f in derived if f[0] in ("Controls", "ActualController", "Alert", "RiskLevel", "BeneficialOwner")]
    return {"as_of": as_of.isoformat(), "effective_case": case_to_dict(effective),
            "documents": list(docs.values()), "identities": list(identities_by_id.values()),
            "excluded_relations": excluded, "retained_relations": retained, "conflicts": conflicts,
            "blocked_rights": blocked_rights, "accepted_rights": accepted, "claims": claims,
            "proofs": [{"fact": f, "lines": engine.explain(f)} for f in claims],
            "input_facts": [{"fact": f.as_tuple(), "source": f.source} for f in facts],
            "derivations": [{"fact": f, "rule": r, "premises": p} for f, (r, p) in engine.why.items()],
            "stats": engine.stats(),
            "equity": economic_equity(effective, payload["target"]),
            "effective_right_targets": sorted(targets),
            "missing_trade_evidence": any(r["key"].startswith("trades:") for r in excluded),
            "note": "身份/凭证为人工声明核验状态，不做外部权威认证；有效期结束日不含当天。"}


def claim_changes(previous: dict | None, current: dict) -> dict:
    before = {tuple(f) for f in previous.get("claims", [])} if previous else set()
    after = {tuple(f) for f in current["claims"]}
    return {"added": sorted(after - before, key=str), "withdrawn": sorted(before - after, key=str),
            "unchanged": len(before & after), "strategy": "rebuild_from_active_evidence",
            "note": "撤回的是当前有效快照不再支持的结论，旧版本证明保留供追溯；不是断言现实命题为假。"}
