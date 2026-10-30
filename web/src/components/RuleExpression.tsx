import { ArrowRight } from "@phosphor-icons/react";

export default function RuleExpression({
  antecedents,
  conclusions,
  guardConditions = [],
  guardComplete = true,
}: {
  antecedents: string[];
  conclusions: string[];
  guardConditions?: string[];
  guardComplete?: boolean;
}) {
  const conditions = [
    ...antecedents.map((text) => ({ text, guard: false })),
    ...guardConditions.map((text) => ({ text, guard: true })),
  ];
  return (
    <div className="rule-expression" aria-label="完整规则公式">
      <div className="rule-premises">
        {conditions.map((condition, index) => (
          <span key={index}>
            {index > 0 && <b aria-label="并且">∧</b>}
            <code className={condition.guard ? "rule-guard" : undefined}>
              {condition.text}
            </code>
          </span>
        ))}
        {!guardComplete && (
          <span className="rule-guard-unavailable">
            {conditions.length > 0 && <b aria-label="并且">∧</b>}
            附加条件尚未展开
          </span>
        )}
      </div>
      <ArrowRight size={18} aria-label="推出" />
      <div className="rule-conclusions">
        {conclusions.map((conclusion, index) => (
          <span key={index}>
            {index > 0 && <b aria-label="并且">∧</b>}
            <code className="consequent">{conclusion}</code>
          </span>
        ))}
      </div>
    </div>
  );
}
