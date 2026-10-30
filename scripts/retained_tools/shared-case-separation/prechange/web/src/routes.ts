import type { LabId } from "./types";
export type Page =
  | "home"
  | "lab"
  | "task"
  | "overview"
  | "entity"
  | "evidence"
  | "edit"
  | "graph"
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
export function parseRoute(hash: string): Address {
  const path = hash.replace(/^#\/?/, "");
  if (!path || path === "home") return { page: "home" };
  if (path === "workbench") return { page: "overview" };
  const parts = path.split("/");
  if (parts[0] === "labs") {
    const labId = parts[1] as LabId;
    if (!Object.hasOwn(modules, labId)) return { page: "notFound" };
    if (parts.length === 2) return { page: "lab", labId };
    if (parts.length === 3 && modules[labId].includes(parts[2]))
      return { page: "task", labId, moduleId: parts[2] };
    return { page: "notFound" };
  }
  let id: string | undefined;
  try {
    id = parts[1] ? decodeURIComponent(parts[1]) : undefined;
  } catch {
    return { page: "notFound" };
  }
  if (parts[0] === "entity" && id && parts.length <= 3) {
    if (parts.length === 3 && !["evidence", "edit"].includes(parts[2]))
      return { page: "notFound" };
    return {
      page:
        parts[2] === "evidence"
          ? "evidence"
          : parts[2] === "edit"
            ? "edit"
            : "entity",
      entityId: id,
    };
  }
  if (parts[0] === "graph" && parts.length <= 2 && (parts.length === 1 || id))
    return { page: "graph", entityId: id };
  if (
    parts.length === 1 &&
    ["settings", "verification", "experiments", "rules", "governance"].includes(
      parts[0],
    )
  )
    return { page: parts[0] as Page };
  return { page: "notFound" };
}
export function labHref(lab: LabId, module?: string) {
  return "#/labs/" + lab + (module ? "/" + module : "");
}
export function entityHref(
  id: string,
  page: "entity" | "evidence" | "edit" = "entity",
) {
  return (
    "#/entity/" + encodeURIComponent(id) + (page === "entity" ? "" : "/" + page)
  );
}
export function graphHref(id?: string) {
  return "#/graph" + (id ? "/" + encodeURIComponent(id) : "");
}
