import type { Analysis, CaseData, GraphEdge } from "./types";
export function draftEdges(data: CaseData): GraphEdge[] {
  return [
    ...data.holdings.map((r, index) => ({
      id: "holding-" + index,
      source: r.holder,
      target: r.target,
      kind: "holding" as const,
      label: (r.ratio * 100).toFixed(1).replace(/\.0$/, "") + "%",
      fact_id: "",
      index,
    })),
    ...data.guarantees.map((r, index) => ({
      id: "guarantee-" + index,
      source: r.guarantor,
      target: r.beneficiary,
      kind: "guarantee" as const,
      label:
        r.amount.toLocaleString("zh-CN", { maximumFractionDigits: 0 }) + "万",
      fact_id: "",
      index,
    })),
    ...data.trades.map((r, index) => ({
      id: "trade-" + index,
      source: r.seller,
      target: r.buyer,
      kind: "trade" as const,
      label:
        r.amount.toLocaleString("zh-CN", { maximumFractionDigits: 0 }) + "万",
      fact_id: "",
      index,
    })),
    ...data.kinships.map((r, index) => ({
      id: "kinship-" + index,
      source: r.a,
      target: r.b,
      kind: "kinship" as const,
      label: r.relation,
      fact_id: "",
      index,
    })),
  ];
}
export function proofOrder(
  analysis: Analysis | null,
  root: string | null,
): string[] {
  if (!analysis || !root) return [];
  const seen = new Set<string>();
  const order: string[] = [];
  const visit = (id: string) => {
    if (seen.has(id) || !analysis.provenance[id]) return;
    seen.add(id);
    for (const p of analysis.provenance[id].premises) visit(p);
    order.push(id);
  };
  visit(root);
  return order;
}
export function proofHighlights(
  analysis: Analysis | null,
  root: string | null,
) {
  const ids = new Set(proofOrder(analysis, root));
  const nodes = new Set<string>();
  const entities = new Set(analysis?.entities.map((e) => e.eid) || []);
  for (const id of ids) {
    for (const value of analysis?.provenance[id]?.fact.slice(1) || [])
      if (typeof value === "string" && entities.has(value)) nodes.add(value);
  }
  const edges = new Set<string>();
  for (const edge of [
    ...(analysis?.edges || []),
    ...(analysis?.derived_edges || []),
  ]) {
    if (ids.has(edge.fact_id)) edges.add(edge.id);
  }
  return { nodes, edges };
}
export function canvasColor(variable: string): string {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 1;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = getComputedStyle(document.documentElement)
    .getPropertyValue(variable)
    .trim();
  ctx.fillRect(0, 0, 1, 1);
  const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
  return "rgb(" + r + "," + g + "," + b + ")";
}
