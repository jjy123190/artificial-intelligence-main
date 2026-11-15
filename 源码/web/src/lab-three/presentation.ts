import type { QueryInfo, Result, Snapshot } from "./types";

export const relations: Record<string, string> = {
  HOLDS: "持股",
  GUARANTEES: "担保",
  TRADES: "交易",
};
export const statuses: Record<string, string> = {
  start: "起点",
  generated: "加入候选",
  expanded: "扩展节点",
  rejected: "专家规则拦截",
  proof_pruned: "已证无解分支",
  rollout: "模拟路径",
  solution: "找到候选路径",
};
export const percent = (n: number) =>
  Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : "—";

export function resultTitle(result: Result["result"]): string {
  if (result.valid) return "已找到可核验路径";
  if (result.found) return "候选路径未通过证据验证";
  return result.status === "budget_exhausted"
    ? "已达到扩展预算"
    : "候选路径已穷尽";
}

export function blockedReason(
  snapshot: Snapshot,
  query: QueryInfo | undefined,
  method: string,
  mode: string,
): string {
  if (!query) return "没有可运行的问题，请重新加载实验数据。";
  if (query.unavailable_methods?.[method])
    return (
      query.unavailable_methods[method] +
      "。请重新生成实验数据，或选择其他方法。"
    );
  if (method !== "hybrid_llm") return "";
  if (mode === "replay" && !query.llm_cached)
    return (
      query.llm_reason ||
      "这个问题没有实测评分。请选择标注“LLM实测”的测试问题。"
    );
  if (mode === "remote" && !snapshot.remote_available)
    return "远程模型尚未配置。请先设置服务环境变量，或使用实测评分重放。";
  return "";
}

export function comparisonCompletion(
  rows: { method: string; result?: unknown; error?: string }[],
  expectedMethods: string[],
) {
  const counts = new Map<string, number>();
  for (const row of rows) counts.set(row.method, (counts.get(row.method) || 0) + 1);
  const attempted = expectedMethods.filter((id) => counts.has(id)).length;
  const successful = expectedMethods.filter((id) =>
    counts.get(id) === 1 && rows.some((row) => row.method === id && row.result != null && !row.error),
  ).length;
  return {
    attempted,
    successful,
    failed: rows.filter((row) => expectedMethods.includes(row.method) && Boolean(row.error)).length,
    complete: successful === expectedMethods.length && rows.length === expectedMethods.length,
  };
}
