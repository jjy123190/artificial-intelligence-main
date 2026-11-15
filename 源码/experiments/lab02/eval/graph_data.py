"""真实走图推理的合成股权样本与源数据扰动。

外部复核标签先采样，随后随机产生相关但不完全一致的业务结构，标签不是规则输出。
结构真值由生成时的业务开关给出；其正确性另用遍历/规则引擎交叉测试。
评测器持有标签和干净图，生产 pipeline 只能接收观测图与采集完整性声明。
"""
from __future__ import annotations
import copy
import math
import random
from dataclasses import dataclass, replace
from core.kb.domain import Entity, Holding, Guarantee, Trade, EquityCase

GRAPH_GENERATOR = {
    "version": 1, "label_prior": .35,
    "shared_control_given_label": (.30, .72),
    "seller_trade_given_label": (.26, .78),
    "holding_cycle_given_label": (.08, .34),
    "guarantee_cycle_given_label": (.12, .50),
    "label_origin": "模拟外部高复核需求标签，先采样；不调用规则引擎生成标签",
    "dependence": "担保负荷与担保圈共享业务开关；不透明度与共同控制交易相关，指标并非条件独立",
    "limits": "合成的3自然人/7企业业务图族，不代表真实企业总体分布，也不覆盖全部27条规则的每个前提组合",
}
SPLIT_SEEDS = {"train": 1103, "validation": 2207, "test": 3313}


@dataclass(frozen=True)
class GraphSample:
    sample_id: str
    case: EquityCase
    target_id: str
    label: int
    oracle_alerts: tuple[str, ...]


@dataclass(frozen=True)
class GraphCondition:
    name: str
    holding_sd: float = 0.0
    amount_log_sd: float = 0.0
    opacity_sd: float = 0.0
    edge_missing: float = 0.0
    spurious_return: float = 0.0

    def __post_init__(self):
        if not self.name:
            raise ValueError("实验条件需要名称")
        for value in (self.holding_sd, self.amount_log_sd, self.opacity_sd, self.edge_missing, self.spurious_return):
            if not math.isfinite(value) or value < 0:
                raise ValueError("扰动参数须为非负有限数")
        if self.edge_missing > 1 or self.spurious_return > 1:
            raise ValueError("缺失/注入概率不能超过1")


GRAPH_CONDITIONS = (
    GraphCondition("clean"),
    GraphCondition("value_noise", .08, .20, .10),
    GraphCondition("edge_missing_020", edge_missing=.20),
    GraphCondition("false_loop_020", spurious_return=.20),
    GraphCondition("combined", .08, .20, .10, .20, .20),
    GraphCondition("severe", .15, .35, .20, .40, .40),
)


@dataclass
class ObservedGraph:
    case: EquityCase
    completeness: dict[str, bool]
    corruption_stats: dict


def generate_graphs(n: int, seed: int, split: str) -> list[GraphSample]:
    if n < 1 or split not in SPLIT_SEEDS:
        raise ValueError("样本数需为正数，split 为 train/validation/test")
    rng = random.Random(seed * 104729 + SPLIT_SEEDS[split])
    samples = []
    for index in range(n):
        sid = f"g{seed}-{split}-{index:06d}"
        label = int(rng.random() < GRAPH_GENERATOR["label_prior"])
        shared = rng.random() < GRAPH_GENERATOR["shared_control_given_label"][label]
        has_trade = rng.random() < GRAPH_GENERATOR["seller_trade_given_label"][label]
        hold_loop = rng.random() < GRAPH_GENERATOR["holding_cycle_given_label"][label]
        guar_loop = rng.random() < GRAPH_GENERATOR["guarantee_cycle_given_label"][label]
        p = [f"{sid}_P{i}" for i in range(3)]
        c = [f"{sid}_C{i}" for i in range(7)]
        opacity_high = rng.random() < min(.95, (.62 if label else .22) + .12 * (shared and has_trade))
        opacity = rng.betavariate(7, 3) if opacity_high else rng.betavariate(2, 9)
        capital = float(rng.randrange(8, 121) * 1000)
        case = EquityCase(label=sid, source=f"合成股权图 v1；seed={seed}；split={split}")
        case.entities.extend(Entity(pid, f"教学自然人{i}", "Person") for i, pid in enumerate(p))
        for i, cid in enumerate(c):
            op = opacity if i == 0 else rng.betavariate(2, 6)
            cap = capital if i == 0 else float(rng.randrange(5, 81) * 1000)
            case.entities.append(Entity(cid, f"教学企业{i}", "Company", "上市公司" if i == 0 else "企业",
                                        cap, op, 1 - op))
        for a, b, ratio in [
            (p[0], c[1], .80), (c[1], c[2], .65), (p[0], c[3], .70),
            (c[2], c[0], .35 if shared else .22), (c[1], c[0], .20 if shared else .10),
            (p[1], c[0], .20 if shared else .54),
            (p[2], c[4], .60), (c[0], c[4], .25), (p[1], c[5], .65), (p[2], c[6], .70),
        ]:
            case.holdings.append(Holding(a, b, ratio))
        if hold_loop:
            case.holdings.append(Holding(c[4], c[0], .12))
        high_load = guar_loop or rng.random() < (.30 if label else .15)
        load = rng.betavariate(6, 3) if high_load else rng.betavariate(2, 10)
        case.guarantees.extend([
            Guarantee(c[0], c[5], round(capital * load, 2)),
            Guarantee(c[5], c[6], round(case.capital_of(c[5]) * rng.uniform(.10, .70), 2)),
        ])
        if guar_loop:
            case.guarantees.append(Guarantee(c[6], c[0], round(case.capital_of(c[6]) * rng.uniform(.10, .70), 2)))
        if has_trade:
            trade_high = rng.random() < (.70 if label else .25)
            share = rng.betavariate(6, 3) if trade_high else rng.betavariate(2, 9)
            case.trades.append(Trade(c[3], c[0], round(capital * share, 2)))
        # 与目标不同控制人的交易不应被误算为关联交易。
        case.trades.append(Trade(c[6], c[0], round(capital * rng.uniform(.02, .25), 2)))
        case.validate()
        alerts = []
        if shared and has_trade:
            alerts.append("关联交易待核查")
        if hold_loop:
            alerts.append("循环持股")
        if guar_loop:
            alerts.append("担保圈")
        samples.append(GraphSample(sid, case, c[0], label, tuple(sorted(alerts))))
    return samples


def assert_graph_splits_disjoint(*splits: list[GraphSample]) -> None:
    ids, entities = set(), set()
    for split in splits:
        current = {s.sample_id for s in split}
        current_entities = {e.eid for s in split for e in s.case.entities}
        if len(current) != len(split) or ids & current or entities & current_entities:
            raise ValueError("图级训练/验证/测试存在样本或实体泄漏")
        ids |= current
        entities |= current_entities


def perturb_graph(case: EquityCase, target_id: str, condition: GraphCondition, seed: int) -> ObservedGraph:
    """只接收源图，不接收标签。金额噪声、持股噪声、删边和错误闭环分别记录。

    数值持股扰动后按被投企业归一化，保证总持股不超过100%，不向模型传入真实比例。
    错误闭环注入是有意的压力条件，不声称反映现实发生频率。
    """
    rng = random.Random(seed)
    observed = copy.deepcopy(case)
    observed.entity(target_id)
    removed = {"holdings": 0, "trades": 0, "guarantees": 0}
    observed.entities = [replace(e, opacity=min(1.0, max(0.0, e.opacity + condition.opacity_sd * rng.gauss(0, 1))))
                         if not e.is_person else e for e in observed.entities]
    holdings = []
    for h in case.holdings:
        drop, noise = rng.random(), rng.gauss(0, 1)
        if drop < condition.edge_missing:
            removed["holdings"] += 1
        else:
            ratio = min(1.0, max(.005, h.ratio + condition.holding_sd * noise))
            holdings.append(replace(h, ratio=ratio))
    totals = {}
    for h in holdings:
        totals[h.target] = totals.get(h.target, 0.0) + h.ratio
    normalized = sum(t > 1 for t in totals.values())
    observed.holdings = [replace(h, ratio=h.ratio / max(1.0, totals[h.target])) for h in holdings]
    for name in ("guarantees", "trades"):
        items = []
        for item in getattr(case, name):
            drop, noise = rng.random(), rng.gauss(0, 1)
            if drop < condition.edge_missing:
                removed[name] += 1
            else:
                items.append(replace(item, amount=item.amount * math.exp(max(-20.0, min(20.0, condition.amount_log_sd * noise)))))
        setattr(observed, name, items)
    injected = 0
    if rng.random() < condition.spurious_return:
        edges = {(g.guarantor, g.beneficiary) for g in observed.guarantees}
        paths = sorted({b for a, middle in edges if a == target_id for source, b in edges
                        if source == middle and b != target_id and (b, target_id) not in edges})
        if paths:
            source = paths[0]
            observed.guarantees.append(Guarantee(source, target_id, observed.capital_of(source) * .4))
            injected = 1
    observed.source += f"；观测条件={condition.name}（不含标签或干净图真值）"
    observed.validate()
    complete = condition.edge_missing == 0
    return ObservedGraph(observed, {"holdings": complete, "trades": complete,
                                   "guarantees": complete, "registry": True},
                         {"removed": removed, "normalized_company_shares": normalized,
                          "injected_guarantee_returns": injected})


def graph_has_cycle_from(edges: list[tuple[str, str]], start: str) -> bool:
    """独立遍历核对回路，用于测试/审计；不是生产规则的替代实现。"""
    adjacent = {}
    for a, b in edges:
        adjacent.setdefault(a, []).append(b)
    queue, seen = list(adjacent.get(start, [])), set()
    while queue:
        node = queue.pop()
        if node == start:
            return True
        if node not in seen:
            seen.add(node)
            queue.extend(adjacent.get(node, []))
    return False
