"""保持目标答案的谓词依赖切片；不声称保留无关谓词的全部结论。"""
from core.kb.rules_dsl import RuleSet


def relevant_rules(rules: RuleSet, goals: set[str]) -> RuleSet:
    needed = set(goals)
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.head_pred() in needed:
                before = len(needed)
                needed.update(p.pred for p in rule.antecedents)
                changed |= len(needed) != before
    return RuleSet([r for r in rules if r.head_pred() in needed])
