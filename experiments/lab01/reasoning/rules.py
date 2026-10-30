"""规则库：股权控制与风险识别的产生式规则（IF 前提 THEN 结论）。"""
from experiments.lab01.kr.production import Rule

RULES = [
    # ---------- 控制关系识别（优先级 3）----------
    Rule("R1", "绝对控股：直接持股超过50%即构成控制",
         [("Holds", "?x", "?y", "?r")],
         [("Controls", "?x", "?y")],
         test=lambda b: b["?r"] > 0.5, priority=3),
    Rule("R2", "教学假设：配偶视为一致行动人（实务须另行取证）",
         [("Kin", "?p", "?q", "?rel")],
         [("Concert", "?p", "?q"), ("Concert", "?q", "?p")],
         test=lambda b: b["?rel"] == "配偶", priority=3),
    Rule("R3", "一致行动人合计持股超过50%，由持股较多者控制",
         [("Concert", "?p", "?q"), ("Holds", "?p", "?c", "?r1"), ("Holds", "?q", "?c", "?r2")],
         [("Controls", "?p", "?c")],
         test=lambda b: b["?r1"] >= b["?r2"] and b["?r1"] + b["?r2"] > 0.5, priority=3),
    Rule("R4", "自身与所控子公司合计持股超过50%即构成控制",
         [("Controls", "?x", "?a"), ("Holds", "?a", "?c", "?r1"), ("Holds", "?x", "?c", "?r2")],
         [("Controls", "?x", "?c")],
         test=lambda b: b["?r1"] + b["?r2"] > 0.5, priority=3),
    Rule("R5", "控制关系具有传递性",
         [("Controls", "?x", "?y"), ("Controls", "?y", "?z")],
         [("Controls", "?x", "?z")],
         test=lambda b: b["?x"] != b["?z"], priority=3),
    Rule("R6", "控制某企业的自然人是其实际控制人",
         [("Controls", "?p", "?c"), ("Person", "?p")],
         [("ActualController", "?c", "?p")], priority=2),

    # ---------- 关联关系 / 可达性（优先级 2）----------
    Rule("R7", "交易双方教学规则下的控制人相同 → 关联交易线索",
         [("Trade", "?a", "?b", "?m"), ("ActualController", "?a", "?p"), ("ActualController", "?b", "?p")],
         [("RelatedTrade", "?a", "?b", "?p")], priority=2),
    Rule("R8", "持股可达（基础）",
         [("Holds", "?a", "?b", "?r")], [("HoldReach", "?a", "?b")], priority=2),
    Rule("R9", "持股可达（传递）",
         [("HoldReach", "?a", "?b"), ("Holds", "?b", "?c", "?r")], [("HoldReach", "?a", "?c")], priority=2),
    Rule("R10", "担保可达（基础）",
         [("Guarantee", "?a", "?b", "?m")], [("GuarReach", "?a", "?b")], priority=2),
    Rule("R11", "担保可达（传递）",
         [("GuarReach", "?a", "?b"), ("Guarantee", "?b", "?c", "?m")], [("GuarReach", "?a", "?c")], priority=2),

    # ---------- 风险预警（优先级 1）----------
    Rule("R12", "上市公司存在共同控制交易 → 待核查预警",
         [("RelatedTrade", "?a", "?b", "?p"), ("IsA", "?b", "上市公司")],
         [("Alert", "?b", "关联交易待核查")], priority=1),
    Rule("R13", "企业经持股链回到自身 → 循环持股预警",
         [("HoldReach", "?a", "?a")], [("Alert", "?a", "循环持股")], priority=1),
    Rule("R14", "企业经担保链回到自身 → 担保圈预警",
         [("GuarReach", "?a", "?a")], [("Alert", "?a", "担保圈")], priority=1),

    # ---------- 风险定级（优先级 0）----------
    Rule("R15", "存在任一预警 → 风险等级为高",
         [("Alert", "?c", "?t")], [("RiskLevel", "?c", "高")], priority=0),
]

RULE_DESC = {r.name: r.desc for r in RULES}

PRED_CN = {
    "Holds": "持股", "Controls": "控制", "Concert": "一致行动", "Kin": "亲属",
    "ActualController": "实际控制人", "Trade": "交易", "RelatedTrade": "关联交易",
    "Guarantee": "担保", "HoldReach": "持股可达", "GuarReach": "担保可达",
    "Alert": "预警", "RiskLevel": "风险等级", "IsA": "属于", "Person": "自然人", "Company": "企业",
}
