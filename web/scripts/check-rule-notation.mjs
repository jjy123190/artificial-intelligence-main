import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";

const source = await readFile(
  new URL("../src/rule-notation.ts", import.meta.url),
  "utf8",
);
const code = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const { holdingExample, formatLiteral, formatPredicate, formatPattern } =
  await import(
    "data:text/javascript;base64," + Buffer.from(code).toString("base64")
  );
const template = {
  name: "R01",
  antecedents: ["Holds(?x,?y,?r)"],
  consequent: "DirectControls(?x,?y)",
  guard_conditions: ["?r > 0.5", "?x ≠ ?y"],
  guard_complete: true,
};
const entities = [
  { eid: "P1", name: "课堂自然人" },
  { eid: "C1", name: "课堂企业" },
];
const record = ["Holds", "P1", "C1", 0.8];
const before = JSON.stringify({ template, entities, record });
const example = holdingExample(template, [record], entities);
assert.equal(example.fact, "Holds(P1,C1,0.8)");
assert.deepEqual(example.bindings, [
  { variable: "?x", value: "P1" },
  { variable: "?y", value: "C1" },
  { variable: "?r", value: "0.8" },
]);
assert.deepEqual(example.antecedents, [
  "Holds(P1,C1,0.8)",
  "0.8 > 0.5",
  "P1 ≠ C1",
]);
assert.equal(example.consequent, "DirectControls(P1,C1)");
assert.ok(example.description.includes("80%"));
assert.equal(JSON.stringify({ template, entities, record }), before);

for (const ratio of [0, 0.2, 0.5, 0.8, 1]) {
  const current = holdingExample(
    template,
    [["Holds", "P1", "C1", ratio]],
    entities,
  );
  assert.equal(current.fact, "Holds(P1,C1," + ratio + ")");
  assert.equal(current.antecedents[1], ratio + " > 0.5");
  assert.equal("proved" in current, false);
  assert.equal("passes" in current, false);
}
assert.equal(holdingExample(template, [], entities), null);
assert.equal(holdingExample(undefined, [record], entities), null);
assert.equal(
  holdingExample({ ...template, guard_complete: false }, [record], entities),
  null,
);
assert.equal(
  holdingExample(
    { ...template, consequent: "Controls(?x,?z)" },
    [record],
    entities,
  ),
  null,
);
for (const ratio of [-0.1, 1.1, NaN, Infinity, "0.8"])
  assert.equal(
    holdingExample(template, [["Holds", "P1", "C1", ratio]], entities),
    null,
  );

const strange = holdingExample(
  template,
  [["Holds", "__proto__", "案例/一?y", 0.8]],
  [
    { eid: "__proto__", name: "保留名称" },
    { eid: "案例/一?y", name: "含标点实体" },
  ],
);
assert.equal(strange.consequent, 'DirectControls(__proto__,"案例/一?y")');
assert.ok(strange.description.includes("保留名称"));
assert.equal(formatLiteral('含"引号'), JSON.stringify('含"引号'));
assert.equal(formatPredicate(["Holds", "P1", "C1", 0.8]), "Holds(P1,C1,0.8)");

const repeated = {
  ...template,
  antecedents: ["Holds(?x,?x,?r)"],
  consequent: "DirectControls(?x,?x)",
  guard_conditions: [],
};
assert.equal(holdingExample(repeated, [record], entities), null);
assert.ok(holdingExample(repeated, [["Holds", "C1", "C1", 0.8]], entities));

assert.equal(formatPattern(["Holds", "?x", "?y", "?r"]), "Holds(?x,?y,?r)");
assert.equal(formatPredicate(["Holds", "?x", "C1", 0.8]), 'Holds("?x",C1,0.8)');
const legacy = {
  ...template,
  name: "R1",
  antecedents: [formatPattern(["Holds", "?x", "?y", "?r"])],
  consequent: formatPattern(["Controls", "?x", "?y"]),
  guard_conditions: ["?r > 0.5"],
};
assert.deepEqual(holdingExample(legacy, [record], entities).antecedents, [
  "Holds(P1,C1,0.8)",
  "0.8 > 0.5",
]);
assert.equal(
  holdingExample(legacy, [record], entities).consequent,
  "Controls(P1,C1)",
);
console.log(
  "Rule notation checks passed: actual current records, complete substitution, no inference or mutation, empty/missing data, boundary proportions, repeated variables and safely escaped identifiers.",
);
