"""企业股权领域的 27 条教学规则，全部为安全的正 Horn 规则。

控制权、共同控制、受益人均为此教学模型下的候选关系，不是法律结论。
允许多种 RiskLevel 同时被推出，展示层取最高等级，不用优先级撤销事实。
"""
from .rules_dsl import (Pattern as P, Rule, RuleSet, all_of, approximately,
                        equals, ge_var, gt, neq, ratio_sum_gt)


def build_rules() -> RuleSet:
    rs = RuleSet()
    def add(name, desc, body, head, guard=None, priority=0, group="", note=""):
        rs.add(Rule(name, desc, tuple(P.of(*p) for p in body), P.of(*head),
                    guard, priority, group, note))

    add("R01", "单方直接持股超过 50% → 绝对控股线索",
        [("Holds", "?x", "?y", "?r")], ("DirectControls", "?x", "?y"),
        all_of(gt("?r", .5), neq("?x", "?y")), 9, "控制关系", "不输出自身控制；不覆盖表决权委托、董事会席位及协议控制。")
    add("R02", "配偶视为一致行动人（教学假设）",
        [("Kin", "?p", "?q", "?rel")], ("Concerts", "?p", "?q"),
        equals("?rel", "配偶"), 9, "控制关系", "配偶并不在现实中无条件等于一致行动人。")
    add("R03", "亲属关系具有公开佐证 → 一致行动候选",
        [("KinEvidenced", "?p", "?q")], ("Concerts", "?p", "?q"),
        priority=8, group="控制关系", note="无佐证的亲属关系不触发本规则。")
    add("R04", "一致行动人合计超过 50% → 持股较多者控制候选",
        [("Concerts", "?p", "?q"), ("Holds", "?p", "?c", "?r1"), ("Holds", "?q", "?c", "?r2")],
        ("DirectControls", "?p", "?c"),
        all_of(ge_var("?r1", "?r2"), ratio_sum_gt(("?r1", "?r2"), .5), neq("?p", "?q")),
        8, "控制关系", "只合计两人；份额相等时保留两个候选，不声称控制人唯一。")
    add("R05", "自身与所控企业的直接持股合计超过 50% → 控制候选",
        [("DirectControls", "?x", "?a"), ("Holds", "?a", "?c", "?r1"), ("Holds", "?x", "?c", "?r2")],
        ("DirectControls", "?x", "?c"),
        all_of(ratio_sum_gt(("?r1", "?r2"), .5), neq("?x", "?a"), neq("?x", "?c")),
        8, "控制关系", "复现 C1 与 C2 合计持有 C3 的 55%；不是多子公司任意集合加总。")
    add("R05b", "所控企业单独持股超过 50% → 控制向上穿透",
        [("DirectControls", "?x", "?a"), ("Holds", "?a", "?c", "?r1")],
        ("DirectControls", "?x", "?c"),
        all_of(gt("?r1", .5), neq("?x", "?a"), neq("?x", "?c")),
        8, "控制关系", "不是缺省否定规则；不假定没有记录就没有直接持股。DirectControls 含穿透控制候选。")
    add("R06", "单条恰好 50% 的持股记录不足以支持绝对控股",
        [("Holds", "?x", "?y", "?r")], ("NotControlsByHalf", "?x", "?y"),
        approximately("?r", .5), 7, "控制关系",
        "该谓词不是 Controls 的逻辑否定；一致行动等其他证据仍可能支持控制。")
    add("R07", "控制传递（不输出自身控制自身）",
        [("Controls", "?x", "?y"), ("Controls", "?y", "?z")], ("Controls", "?x", "?z"),
        neq("?x", "?z"), 7, "控制关系")
    add("R08", "将 DirectControls 纳入 Controls",
        [("DirectControls", "?x", "?y")], ("Controls", "?x", "?y"),
        priority=6, group="控制关系")
    add("R09", "控制企业的自然人 → 实际控制人候选",
        [("Controls", "?p", "?c"), ("Person", "?p"), ("Company", "?c")],
        ("ActualController", "?c", "?p"), priority=6, group="控制关系")

    add("R10", "持股可达：基础边", [("Holds", "?a", "?b", "?r")],
        ("HoldReach", "?a", "?b"), priority=5, group="结构可达")
    add("R11", "持股可达：传递闭包（保留回到自身的路径）",
        [("HoldReach", "?a", "?b"), ("Holds", "?b", "?c", "?r")],
        ("HoldReach", "?a", "?c"), priority=5, group="结构可达",
        note="不能排除 ?a=?c，否则循环持股预警永远无法触发。")
    add("R12", "担保可达：基础边", [("Guarantee", "?a", "?b", "?m")],
        ("GuarReach", "?a", "?b"), priority=5, group="结构可达")
    add("R13", "担保可达：传递闭包（保留回路）",
        [("GuarReach", "?a", "?b"), ("Guarantee", "?b", "?c", "?m")],
        ("GuarReach", "?a", "?c"), priority=5, group="结构可达")

    add("R14", "同一自然人控制的不同企业 → 共同控制关系",
        [("ActualController", "?a", "?p"), ("ActualController", "?b", "?p")],
        ("SharedController", "?a", "?b"), neq("?a", "?b"), 4, "关联交易")
    add("R15", "共同控制企业之间的实际交易 → 关联交易线索",
        [("SharedController", "?a", "?b"), ("Trade", "?a", "?b", "?m")],
        ("RelatedTrade", "?a", "?b"), priority=4, group="关联交易")
    add("R16", "共同控制关系反向匹配交易（等价冗余规则，用于去冗余实验）",
        [("SharedController", "?a", "?b"), ("Trade", "?b", "?a", "?m")],
        ("RelatedTrade", "?b", "?a"), priority=4, group="关联交易")
    add("R17", "存在三层及以上简单持股路径 → 多层结构线索",
        [("ElevatedReach", "?a", "?b")], ("OpaqueLayer", "?a", "?b"),
        priority=3, group="不透明线索", note="输入由路径预处理得到（上限8层）；多层不等于违法。")
    add("R18", "持有低资本且未见交易记录的企业 → 壳结构候选",
        [("Holds", "?x", "?s", "?r"), ("Shell", "?s")], ("OpaqueShell", "?x", "?s"),
        priority=3, group="不透明线索", note="注册资本≤500万元且样本无交易只是代理，不能认定真实空壳。")

    add("R19", "上市买方存在关联交易线索 → 待核查预警",
        [("RelatedTrade", "?a", "?b"), ("Listed", "?b")],
        ("Alert", "?b", "关联交易待核查"), priority=2, group="风险预警")
    add("R19b", "上市卖方存在关联交易线索 → 待核查预警",
        [("RelatedTrade", "?a", "?b"), ("Listed", "?a")],
        ("Alert", "?a", "关联交易待核查"), priority=2, group="风险预警")
    add("R20", "持股路径回到自身 → 循环持股线索",
        [("HoldReach", "?a", "?a")], ("Alert", "?a", "循环持股"), priority=2, group="风险预警")
    add("R21", "担保路径回到自身 → 担保圈线索",
        [("GuarReach", "?a", "?a")], ("Alert", "?a", "担保圈"), priority=2, group="风险预警")
    add("R22", "有预警 → 高复核优先级",
        [("Alert", "?c", "?t")], ("RiskLevel", "?c", "高"), priority=1, group="风险定级")
    add("R23", "多层持股路径的目标企业 → 中复核优先级",
        [("OpaqueLayer", "?x", "?c"), ("Company", "?c")],
        ("RiskLevel", "?c", "中"), priority=1, group="风险定级",
        note="不含否定前提；若高、中同时出现，展示层取高。")
    add("R23b", "壳结构候选企业 → 中复核优先级",
        [("OpaqueShell", "?x", "?c"), ("Company", "?c")],
        ("RiskLevel", "?c", "中"), priority=1, group="风险定级")
    add("R24", "可达企业的自然人股东 → 受益人候选（非最终认定）",
        [("HoldReach", "?p", "?c"), ("Person", "?p"), ("Company", "?c")],
        ("BeneficialOwner", "?c", "?p"), priority=1, group="受益人候选",
        note="未计算穿透份额和实际收益；不能称为已确认的最终受益人。")
    return rs


PRED_CN = {
    "Person": "自然人", "Company": "企业", "Listed": "上市公司", "StateOwned": "国有企业",
    "Holds": "持股", "Guarantee": "担保", "Trade": "交易", "Kin": "亲属",
    "KinEvidenced": "亲属佐证", "Shell": "壳结构代理", "Concerts": "一致行动候选",
    "DirectControls": "控制基础候选", "Controls": "控制", "ActualController": "实控人候选",
    "HoldReach": "持股可达", "GuarReach": "担保可达", "ElevatedReach": "多层路径预处理",
    "SharedController": "共同控制", "RelatedTrade": "关联交易线索", "OpaqueLayer": "多层结构",
    "OpaqueShell": "壳结构线索", "NotControlsByHalf": "50%边界记录", "Alert": "待核查预警",
    "RiskLevel": "复核优先级", "BeneficialOwner": "受益人候选",
}
DERIVED_PREDS = {r.head_pred() for r in build_rules()}
BASE_PREDS = {p.pred for r in build_rules() for p in r.antecedents} - DERIVED_PREDS


def describe_rules(rs: RuleSet | None = None) -> str:
    rs = rs if rs is not None else build_rules()
    return f"规则库共 {len(rs)} 条\n\n{rs.describe()}"
