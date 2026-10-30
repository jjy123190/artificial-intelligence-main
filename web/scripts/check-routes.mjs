import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";
const source = await readFile(
  new URL("../src/routes.ts", import.meta.url),
  "utf8",
);
const code = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const {
  parseRoute,
  entityHref,
  graphHref,
  labHref,
  caseHref,
  analysisHref,
  reasoningGraphHref,
  settingsHref,
  recordsHref,
  rulesHref,
  governanceHref,
  experimentResultsHref,
  modules,
  analysisPages,
} = await import(
  "data:text/javascript;base64," + Buffer.from(code).toString("base64")
);
const lab2 = (page, moduleId, entityId) => ({
  page,
  labId: "lab02",
  moduleId,
  ...(entityId === undefined ? {} : { entityId }),
});
const cases = [
  ["", { page: "home" }],
  ["#/home", { page: "home" }],
  ["#/workbench", { page: "overview" }],
  ["#/labs/lab01", { page: "lab", labId: "lab01" }],
  ["#/labs/lab02", { page: "lab", labId: "lab02" }],
  ["#/case/C3", { page: "caseEntity", entityId: "C3" }],
  ["#/graph", { page: "graph", entityId: undefined }],
  ["#/graph/C3", { page: "graph", entityId: "C3" }],
  [analysisHref(), lab2("analysis", "hybrid")],
  [settingsHref(), lab2("settings", "hybrid")],
  [recordsHref(), lab2("verification", "deterministic")],
  [reasoningGraphHref(), lab2("reasoningGraph", "deterministic")],
  [experimentResultsHref(), lab2("experiments", "optimization")],
  [rulesHref(), lab2("rules", "modeling")],
  [governanceHref(), lab2("governance", "modeling")],
  ["#/entity/C3", lab2("entity", "hybrid", "C3")],
  ["#/entity/C6/evidence", lab2("evidence", "deterministic", "C6")],
  ["#/entity/C3/edit", lab2("edit", "modeling", "C3")],
  ...[
    ["settings", "hybrid"],
    ["verification", "deterministic"],
    ["experiments", "optimization"],
    ["rules", "modeling"],
    ["governance", "modeling"],
  ].map(([page, moduleId]) => ["#/" + page, lab2(page, moduleId)]),
];
const invalid = [
  "#/materials",
  "#/labs/lab01/deliverables",
  "#/missing",
  "#/case",
  "#/case/",
  "#/case/C3/more",
  "#/case/%INVALID",
  "#/workbench/more",
  "#/labs/lab99",
  "#/labs/__proto__",
  "#/labs/constructor/demo",
  "#/labs/lab01/missing",
  "#/labs/lab02/uncertain/extra",
  "#/labs/lab01/demo/entity/C3",
  "#/labs/lab02/hybrid/entity/",
  "#/labs/lab02/hybrid/entity/C3/evidence",
  "#/labs/lab02/modeling/entity/C3",
  "#/labs/lab02/deterministic/entity/C3/edit",
  "#/labs/lab02/modeling/rules/extra",
  "#/labs/lab02/deterministic/graph/",
  "#/labs/lab02/deterministic/graph/C3/more",
  "#/labs/lab02/hybrid/records",
  "#/entity",
  "#/entity/C3/unknown",
  "#/entity/C3/edit/more",
  "#/graph/",
  "#/graph/C3/more",
  "#/settings/extra",
  "#/entity/%INVALID",
  "#/__proto__",
  "#/constructor",
];
for (const [hash, expected] of [
  ...cases,
  ...invalid.map((hash) => [hash, { page: "notFound" }]),
])
  assert.deepEqual(parseRoute(hash), expected, hash);
assert.deepEqual(modules.lab01, [
  "representation",
  "industry",
  "demo",
  "limitations",
]);
assert.deepEqual(modules.lab02, [
  "modeling",
  "deterministic",
  "uncertain",
  "optimization",
  "hybrid",
]);
for (const [lab, tasks] of Object.entries(modules)) {
  assert.deepEqual(parseRoute(labHref(lab)), { page: "lab", labId: lab });
  for (const moduleId of tasks)
    assert.deepEqual(parseRoute(labHref(lab, moduleId)), {
      page: "task",
      labId: lab,
      moduleId,
    });
}
for (const id of [
  "C3",
  "教学编号/一",
  "id with spaces",
  "x%y",
  "包含#符号",
  "__proto__",
]) {
  assert.deepEqual(parseRoute(caseHref(id)), {
    page: "caseEntity",
    entityId: id,
  });
  assert.deepEqual(parseRoute(graphHref(id)), { page: "graph", entityId: id });
  assert.deepEqual(
    parseRoute(reasoningGraphHref(id)),
    lab2("reasoningGraph", "deterministic", id),
  );
  for (const view of ["entity", "evidence", "edit"]) {
    const moduleId =
      view === "evidence"
        ? "deterministic"
        : view === "edit"
          ? "modeling"
          : "hybrid";
    assert.deepEqual(
      parseRoute(entityHref(id, view)),
      lab2(view, moduleId, id),
    );
    assert.ok(entityHref(id, view).startsWith("#/labs/lab02/"));
  }
}
for (const raw of [
  "#/workbench",
  caseHref("C3"),
  graphHref(),
  graphHref("C3"),
]) {
  assert.equal(parseRoute(raw).labId, undefined);
  assert.equal(analysisPages.includes(parseRoute(raw).page), false);
}
console.log(
  "Route checks passed: raw-data public views, correctly scoped experiment-two results, legacy aliases, encoded identifiers and strict invalid routes.",
);
