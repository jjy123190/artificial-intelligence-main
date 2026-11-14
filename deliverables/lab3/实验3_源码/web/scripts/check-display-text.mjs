import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import ts from "typescript";

const source = await readFile(new URL("../src/display-text.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
});
const { displayText, displayJson } = await import(
  "data:text/javascript;base64," + Buffer.from(outputText).toString("base64")
);
const samples = [
  ["教学投资人1", "投资人1"],
  ["从教学投资人2出发，教学自然人提供证据。", "从投资人2出发，自然人提供证据。"],
  ["教学预设", "手工预设"],
  ["教学合成图", "数据图谱"],
  ["共享教学数据", "共享数据"],
  ["教学假设：配偶视为一致行动人", "规则假设：配偶视为一致行动人"],
  ["合成数据 / 合成标签", "数据 / 标签"],
  ["全部分类标签均为合成标签，不代表真实企业监管准确率。", "分类指标基于当前数据计算。"],
  ["第一次课堂案例的本地数据快照", "第一次实验案例的本地数据快照"],
  ["MCTS路径模拟；D-S证据合成；奖励0.55", "MCTS路径模拟；D-S证据合成；奖励0.55"],
  ["模型解释未通过完整证据认证", "模型解释未通过完整证据认证"],
  ["test-7-0-Q0 / hybrid_llm / 128 / 96.8%", "test-7-0-Q0 / hybrid_llm / 128 / 96.8%"],
  [null, ""],
  [undefined, ""],
];
for (const [input, expected] of samples) {
  assert.equal(displayText(input), expected);
  assert.equal(displayText(displayText(input)), expected, "idempotent display conversion");
}
const raw = Object.freeze({
  id: "test-7-0-Q0", name: "教学投资人1", score: 0.55, valid: false,
  graph_sha256: "9f2ba13f", edge_ids: Object.freeze(["test-7-0-E1"]),
  proof: Object.freeze({ text: "教学投资人1 --HOLDS--> 企业01", grounded: false }),
});
const before = JSON.stringify(raw);
const preview = JSON.parse(displayJson(raw));
assert.equal(preview.name, "投资人1");
assert.equal(preview.proof.text, "投资人1 --HOLDS--> 企业01");
assert.equal(preview.score, raw.score);
assert.equal(preview.valid, raw.valid);
assert.equal(preview.proof.grounded, false);
assert.equal(preview.graph_sha256, raw.graph_sha256);
assert.deepEqual(preview.edge_ids, raw.edge_ids);
assert.equal(JSON.stringify(raw), before, "rendering must not mutate input or export evidence");
assert.equal(displayJson(undefined), "");
assert.equal(displayJson(null), "null");

// Static JSX must use neutral wording; the helper itself retains legacy match patterns.
const forbidden = /教学|合成数据|合成标签|合成指标样本|合成设定|课堂演示/;
async function checkStatic(folder) {
  for (const entry of await readdir(folder, { withFileTypes: true })) {
    const url = new URL(entry.name + (entry.isDirectory() ? "/" : ""), folder);
    if (entry.isDirectory()) await checkStatic(url);
    else if (entry.name.endsWith(".tsx")) {
      assert.doesNotMatch(await readFile(url, "utf8"), forbidden, url.pathname);
    }
  }
}
await checkStatic(new URL("../src/", import.meta.url));
console.log("Website wording checks passed: aliases, algorithm terminology, metrics, immutable evidence and static copy.");
