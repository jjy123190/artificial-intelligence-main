import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";
async function load(relative) {
  const source = await readFile(new URL(relative, import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
  });
  return import(
    "data:text/javascript;base64," + Buffer.from(outputText).toString("base64")
  );
}
const { percent, resultTitle, blockedReason, comparisonCompletion } = await load(
  "../src/lab-three/presentation.ts",
);
const { treeStates, stateKey } = await load("../src/lab-three/tree-model.ts");
assert.equal(percent(Number.NaN), "—");
assert.equal(percent(0), "0.0%");
assert.equal(resultTitle({ valid: true }), "已找到可核验路径");
assert.equal(
  resultTitle({ found: true, valid: false }),
  "候选路径未通过证据验证",
);
assert.equal(resultTitle({ status: "exhausted" }), "候选路径已穷尽");
assert.equal(resultTitle({ status: "budget_exhausted" }), "已达到扩展预算");
const query = {
  llm_cached: false,
  llm_reason: "缺少实测记录",
  unavailable_methods: {},
};
assert.equal(blockedReason({}, query, "bfs", "replay"), "");
assert.match(blockedReason({}, query, "hybrid_llm", "replay"), /实测/);
assert.match(blockedReason({}, query, "hybrid_llm", "remote"), /未配置/);
assert.equal(
  blockedReason({ remote_available: true }, query, "hybrid_llm", "remote"),
  "",
);
assert.match(
  blockedReason(
    {},
    { ...query, unavailable_methods: { hybrid: "模型缺失" } },
    "hybrid",
    "replay",
  ),
  /模型缺失/,
);
const makeStep = (id, edges, status = "generated") => ({
  id,
  node: "n" + edges.length,
  path: ["root", ...edges.map((_, i) => "n" + (i + 1))],
  edge_ids: edges,
  parent_path: edges.slice(0, -1),
  depth: edges.length,
  g: edges.length,
  h: null,
  f: null,
  p: null,
  status,
});
const trace = [
  makeStep(0, [], "start"),
  ...Array.from({ length: 150 }, (_, i) => makeStep(i + 1, ["e" + i])),
  makeStep(152, ["e149", "leaf"], "solution"),
];
const state = treeStates(trace, trace.length - 1, 120);
assert.equal(state.states.length, 120);
assert.equal(state.hidden, 32);
assert.ok(
  state.states.some((s) => stateKey(s.edge_ids) === stateKey(["e149", "leaf"])),
);
assert.ok(
  state.states.some((s) => stateKey(s.edge_ids) === stateKey(["e149"])),
);
assert.notEqual(stateKey(["a/b", "c"]), stateKey(["a", "b/c"]));
assert.equal(treeStates(trace, 0).states.length, 1);
assert.equal(treeStates([], 0).states.length, 0);
const omitted = treeStates([makeStep(1701, ["e1", "e2", "e3"], "solution")], 0);
assert.equal(omitted.states.length, 4);
assert.equal(omitted.states.filter((s) => s.status === "path").length, 3);
console.log(
  "Lab03 presentation checks passed: distinct outcomes, missing-model gates, null metrics, focused-branch windowing and missing ancestor reconstruction.",
);

const compared = [{ method: "bfs", result: {} }, { method: "mcts", error: "timeout" }];
assert.equal(comparisonCompletion(compared, ["bfs", "mcts"]).complete, false);
assert.equal(comparisonCompletion(compared, ["bfs", "mcts"]).successful, 1);
assert.equal(comparisonCompletion(compared.slice(0, 1), ["bfs", "mcts"]).complete, false);
assert.equal(comparisonCompletion([{method:"bfs",result:{}},{method:"mcts",result:{}}], ["bfs", "mcts"]).complete, true);
assert.equal(comparisonCompletion([{method:"bfs",result:{}},{method:"bfs",result:{}}], ["bfs", "mcts"]).complete, false);
console.log("Comparison exports distinguish complete, failed and cancelled runs.");
