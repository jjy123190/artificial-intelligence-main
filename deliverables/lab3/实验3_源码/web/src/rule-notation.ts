import type { RuleItem } from "./types";

export type RuleTemplate = Pick<
  RuleItem,
  "name" | "antecedents" | "consequent" | "guard_conditions" | "guard_complete"
>;

export type RuleExample = {
  rule: string;
  fact: string;
  description: string;
  bindings: { variable: string; value: string }[];
  antecedents: string[];
  consequent: string;
};

const variablePattern = /\?[A-Za-z][A-Za-z0-9_]*/g;

export function formatLiteral(value: string | number): string {
  if (typeof value === "number") return String(value);
  return /^[\p{L}_][\p{L}\p{N}_]*$/u.test(value)
    ? value
    : JSON.stringify(value);
}

export function formatPredicate(fact: (string | number)[]): string {
  return fact.length > 1
    ? String(fact[0]) + "(" + fact.slice(1).map(formatLiteral).join(",") + ")"
    : String(fact[0] ?? "");
}

/** Rule patterns preserve variable markers; concrete facts quote string literals. */
export function formatPattern(pattern: (string | number)[]): string {
  const formatTerm = (value: string | number) =>
    typeof value === "string" && /^\?[A-Za-z][A-Za-z0-9_]*$/.test(value)
      ? value
      : formatLiteral(value);
  return pattern.length > 1
    ? String(pattern[0]) +
        "(" +
        pattern.slice(1).map(formatTerm).join(",") +
        ")"
    : String(pattern[0] ?? "");
}

/** Substitute one actual record into a template, without evaluating or inferring. */
export function holdingExample(
  template: RuleTemplate | undefined,
  facts: (string | number)[][],
  entities: { eid: string; name: string }[],
): RuleExample | null {
  if (!template?.guard_complete || template.antecedents.length !== 1)
    return null;
  const pattern = template.antecedents[0].match(/^Holds\(([^()]*)\)$/);
  const variables = pattern?.[1].split(",").map((value) => value.trim());
  if (
    !variables ||
    variables.length !== 3 ||
    variables.some((value) => !/^\?[A-Za-z][A-Za-z0-9_]*$/.test(value))
  )
    return null;
  const fact = facts.find(
    (row) =>
      row.length === 4 &&
      row[0] === "Holds" &&
      typeof row[1] === "string" &&
      typeof row[2] === "string" &&
      typeof row[3] === "number" &&
      Number.isFinite(row[3]) &&
      row[3] >= 0 &&
      row[3] <= 1 &&
      variables.every((variable, index) =>
        variables.every(
          (other, otherIndex) =>
            variable !== other || row[index + 1] === row[otherIndex + 1],
        ),
      ),
  );
  if (!fact) return null;
  const binding = new Map(
    variables.map((variable, index) => [variable, fact[index + 1]]),
  );
  const expressions = [
    ...template.antecedents,
    ...template.guard_conditions,
    template.consequent,
  ];
  if (
    expressions.some((expression) =>
      Array.from(expression.matchAll(variablePattern)).some(
        ([variable]) => !binding.has(variable),
      ),
    )
  )
    return null;
  const substitute = (expression: string) =>
    expression.replace(variablePattern, (variable) =>
      formatLiteral(binding.get(variable)!),
    );
  const names = new Map(entities.map((entity) => [entity.eid, entity.name]));
  const holder = String(fact[1]);
  const target = String(fact[2]);
  const ratio = Number(fact[3]);
  return {
    rule: template.name,
    fact: formatPredicate(fact),
    description:
      (names.get(holder) || holder) +
      "（" +
      holder +
      "）持有 " +
      (names.get(target) || target) +
      "（" +
      target +
      "）" +
      (ratio * 100).toLocaleString("zh-CN", { maximumFractionDigits: 4 }) +
      "% 股权。",
    bindings: Array.from(binding, ([variable, value]) => ({
      variable,
      value: formatLiteral(value),
    })),
    antecedents: [...template.antecedents, ...template.guard_conditions].map(
      substitute,
    ),
    consequent: substitute(template.consequent),
  };
}
