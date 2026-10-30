import type { LabId } from "./types";
export type Page =
  | "home"
  | "lab"
  | "task"
  | "overview"
  | "caseEntity"
  | "analysis"
  | "entity"
  | "evidence"
  | "edit"
  | "graph"
  | "reasoningGraph"
  | "settings"
  | "verification"
  | "experiments"
  | "rules"
  | "governance"
  | "notFound";
export type Address = {
  page: Page;
  entityId?: string;
  labId?: LabId;
  moduleId?: string;
};
export const modules: Record<LabId, string[]> = {
  lab01: ["representation", "industry", "demo", "limitations"],
  lab02: ["modeling", "deterministic", "uncertain", "optimization", "hybrid"],
};
export const analysisPages: Page[] = [
  "analysis",
  "entity",
  "evidence",
  "edit",
  "reasoningGraph",
  "settings",
  "verification",
];
const experimentViews: Record<string, [Page, string]> = {
  settings: ["settings", "hybrid"],
  verification: ["verification", "deterministic"],
  experiments: ["experiments", "optimization"],
  rules: ["rules", "modeling"],
  governance: ["governance", "modeling"],
};
const experimentAddress = (
  page: Page,
  moduleId: string,
  entityId?: string,
): Address => ({
  page,
  labId: "lab02",
  moduleId,
  ...(entityId === undefined ? {} : { entityId }),
});
export function parseRoute(hash: string): Address {
  const route = hash.replace(/^#\/?/, "");
  if (!route || route === "home") return { page: "home" };
  if (route === "workbench") return { page: "overview" };
  let parts: string[];
  try {
    parts = route.split("/").map(decodeURIComponent);
  } catch {
    return { page: "notFound" };
  }
  if (parts[0] === "labs") {
    const labId = parts[1] as LabId;
    if (!Object.hasOwn(modules, labId)) return { page: "notFound" };
    if (parts.length === 2) return { page: "lab", labId };
    const moduleId = parts[2];
    if (!modules[labId].includes(moduleId)) return { page: "notFound" };
    if (parts.length === 3) return { page: "task", labId, moduleId };
    if (labId !== "lab02") return { page: "notFound" };
    if (parts.length === 4) {
      if (moduleId === "hybrid" && parts[3] === "overview")
        return experimentAddress("analysis", moduleId);
      if (moduleId === "hybrid" && parts[3] === "settings")
        return experimentAddress("settings", moduleId);
      if (moduleId === "deterministic" && parts[3] === "records")
        return experimentAddress("verification", moduleId);
      if (moduleId === "deterministic" && parts[3] === "graph")
        return experimentAddress("reasoningGraph", moduleId);
      if (moduleId === "optimization" && parts[3] === "results")
        return experimentAddress("experiments", moduleId);
      if (moduleId === "modeling" && parts[3] === "rules")
        return experimentAddress("rules", moduleId);
      if (moduleId === "modeling" && parts[3] === "governance")
        return experimentAddress("governance", moduleId);
    }
    if (
      moduleId === "deterministic" &&
      parts[3] === "graph" &&
      parts.length === 5 &&
      parts[4]
    )
      return experimentAddress("reasoningGraph", moduleId, parts[4]);
    if (parts[3] === "entity" && parts[4]) {
      if (moduleId === "hybrid" && parts.length === 5)
        return experimentAddress("entity", moduleId, parts[4]);
      if (
        moduleId === "deterministic" &&
        parts.length === 6 &&
        parts[5] === "evidence"
      )
        return experimentAddress("evidence", moduleId, parts[4]);
      if (moduleId === "modeling" && parts.length === 6 && parts[5] === "edit")
        return experimentAddress("edit", moduleId, parts[4]);
    }
    return { page: "notFound" };
  }
  const id = parts[1];
  if (parts[0] === "case" && parts.length === 2 && id)
    return { page: "caseEntity", entityId: id };
  // Previous links remain usable, but inference pages always belong to experiment two.
  if (parts[0] === "entity" && id && parts.length <= 3) {
    if (parts.length === 2) return experimentAddress("entity", "hybrid", id);
    if (parts[2] === "evidence")
      return experimentAddress("evidence", "deterministic", id);
    if (parts[2] === "edit") return experimentAddress("edit", "modeling", id);
    return { page: "notFound" };
  }
  if (parts[0] === "graph" && parts.length <= 2 && (parts.length === 1 || id))
    return { page: "graph", entityId: id };
  if (parts.length === 1 && Object.hasOwn(experimentViews, parts[0])) {
    const [page, moduleId] = experimentViews[parts[0]];
    return experimentAddress(page, moduleId);
  }
  return { page: "notFound" };
}
export function labHref(lab: LabId, module?: string) {
  return "#/labs/" + lab + (module ? "/" + module : "");
}
export const analysisHref = () => labHref("lab02", "hybrid") + "/overview";
export const caseHref = (id: string) => "#/case/" + encodeURIComponent(id);
export function entityHref(
  id: string,
  page: "entity" | "evidence" | "edit" = "entity",
) {
  const moduleId =
    page === "evidence"
      ? "deterministic"
      : page === "edit"
        ? "modeling"
        : "hybrid";
  return (
    labHref("lab02", moduleId) +
    "/entity/" +
    encodeURIComponent(id) +
    (page === "entity" ? "" : "/" + page)
  );
}
export function graphHref(id?: string) {
  return "#/graph" + (id ? "/" + encodeURIComponent(id) : "");
}
export function reasoningGraphHref(id?: string) {
  return (
    labHref("lab02", "deterministic") +
    "/graph" +
    (id ? "/" + encodeURIComponent(id) : "")
  );
}
export const settingsHref = () => labHref("lab02", "hybrid") + "/settings";
export const recordsHref = () => labHref("lab02", "deterministic") + "/records";
export const experimentResultsHref = () =>
  labHref("lab02", "optimization") + "/results";
export const rulesHref = () => labHref("lab02", "modeling") + "/rules";
export const governanceHref = () =>
  labHref("lab02", "modeling") + "/governance";
