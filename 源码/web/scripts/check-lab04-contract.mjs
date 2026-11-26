/** Pure contract checks: failures must not enter charts or paired results. */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";
const source = await readFile(
  new URL("../src/lab-four/presentation.ts", import.meta.url),
  "utf8",
);
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
});
const { validEpisode, sameEpisode, signedPoints } = await import(
  "data:text/javascript;base64," +
    Buffer.from(
      outputText + "\n//# sourceURL=lab04-presentation-contract.js",
    ).toString("base64")
);
const classes = Array.from({ length: 5 }, (_, i) => ({
  id: `c${i}`,
  name: `类别${i}`,
}));
const result = {
  compute: "live",
  method: "calibrated",
  calibration: "full",
  shots: 1,
  way: 5,
  seed: 42,
  accuracy: 1,
  correct: 75,
  total: 75,
  latency_ms: 10,
  classes,
  support: classes.map((c, i) => ({
    id: `s${i}`,
    class_id: c.id,
    image: "data:",
  })),
  queries: Array.from({ length: 75 }, (_, i) => {
    const c = i % 5;
    const scores = Array(5).fill(0.05);
    scores[c] = 0.8;
    return {
      id: `q${i}`,
      target_id: `c${c}`,
      predicted_id: `c${c}`,
      confidence: 0.8,
      probabilities: scores,
      image: "data:",
    };
  }),
  pool_hash: "a".repeat(64),
  artifact_revision: "b".repeat(64),
  study_id: "v2",
  transductive: true,
  balanced_queries: true,
  checkpoint_sha256: "c".repeat(64),
  scope: "Test fixture; not measured benchmark results",
  metric: "class_conditional_distance",
  feature_motion: "unchanged",
  prototype_role: "decision_prototypes",
  config: {},
  steps: [{ name: "Test fixture", detail: "Pure contract validation" }],
  projection: {
    method: "PCA",
    explained_variance: [0.3, 0.2],
    feature_dimensions: 64,
    computed_after_prediction: true,
  },
  prototypes: classes.map((c, i) => ({
    class_id: c.id,
    before: [i / 5, 0.5],
    after: [i / 5, 0.5],
  })),
};
result.points = [
  ...result.support.map((s) => ({
    id: s.id,
    class_id: s.class_id,
    kind: "support",
    before: [0.1, 0.2],
    after: [0.1, 0.2],
  })),
  ...result.queries.map((q) => ({
    id: q.id,
    class_id: q.target_id,
    kind: "query",
    before: [0.2, 0.3],
    after: [0.2, 0.3],
  })),
];
assert.equal(validEpisode(result), true);
assert.equal(sameEpisode(result, structuredClone(result)), true);
for (const malformed of [null, undefined, false, 1, "fixture", [], {}]) {
  assert.equal(
    validEpisode(malformed),
    false,
    "non-object/incomplete response",
  );
}
for (const mutate of [
  (r) => (r.accuracy = 0.9),
  (r) => (r.queries[0].probabilities[0] = NaN),
  (r) => (r.queries[0].confidence = 1),
  (r) => (r.queries[0].id = r.support[0].id),
  (r) => (r.queries[0].id = r.queries[1].id),
  (r) => (r.seed = -1),
  (r) => (r.classes[0].id = r.classes[1].id),
  (r) => (r.queries[0].target_id = "bad"),
  (r) => (r.decision_rule = "arbitrary"),
  (r) => (r.queries[0].probabilities = [0.5, 0.5]),
  (r) => (r.latency_ms = NaN),
  (r) => (r.total = 70),
  (r) => (r.classes[0] = null),
  (r) => (r.classes[0].name = {}),
  (r) => (r.support[0] = null),
  (r) => (r.support[0].id = ""),
  (r) => (r.support[0].image = {}),
  (r) => (r.support[0].class_id = r.classes[1].id),
  (r) => (r.queries[0] = null),
  (r) => delete r.queries[0].id,
  (r) => (r.queries[0].image = ""),
  (r) => (r.queries[0].confidence = null),
  (r) => (r.queries[0].probabilities[0] = "0.8"),
  (r) => (r.points = null),
  (r) => (r.points[0] = null),
  (r) => (r.points[0].before = [NaN, 0]),
  (r) => (r.points[0].after = [0, Infinity]),
  (r) => (r.points[0].before = [0, 1, 2]),
  (r) => (r.points[0].id = r.points[1].id),
  (r) => (r.points[0].class_id = r.classes[1].id),
  (r) => (r.points[0].kind = "query"),
  (r) => r.points.pop(),
  (r) => (r.prototypes = null),
  (r) => (r.prototypes[0] = null),
  (r) => (r.prototypes[0].class_id = r.classes[1].id),
  (r) => (r.prototypes[0].before = [0, NaN]),
  (r) => r.prototypes.pop(),
  (r) => (r.steps = null),
  (r) => (r.steps[0] = null),
  (r) => (r.steps[0].detail = {}),
  (r) => (r.steps = []),
  (r) => (r.scope = {}),
  (r) => (r.checkpoint_sha256 = "not-a-sha256"),
  (r) => (r.pool_hash = ""),
  (r) => (r.artifact_revision = ""),
  (r) => (r.study_id = {}),
  (r) => (r.method = "invented"),
  (r) => (r.calibration = "toString"),
  (r) => (r.method = "baseline"),
  (r) => (r.transductive = "yes"),
  (r) => (r.transductive = false),
  (r) => (r.balanced_queries = "yes"),
  (r) => (r.metric = {}),
  (r) => (r.feature_motion = "invented"),
  (r) => (r.prototype_role = "invented"),
  (r) => (r.config = []),
  (r) => (r.projection = null),
  (r) => (r.projection.method = "invented"),
  (r) => (r.projection.explained_variance = null),
  (r) => (r.projection.explained_variance = [NaN, 0.2]),
  (r) => (r.projection.explained_variance = [0.9, 0.9]),
  (r) => (r.projection.explained_variance = [0.2]),
  (r) => (r.projection.feature_dimensions = 0),
  (r) => (r.projection.computed_after_prediction = false),
]) {
  const value = structuredClone(result);
  mutate(value);
  assert.equal(validEpisode(value), false);
}
for (const mutate of [
  (r) => r.seed++,
  (r) => (r.pool_hash = "other"),
  (r) => (r.artifact_revision = "other"),
  (r) => r.support.reverse(),
  (r) => r.queries.reverse(),
  (r) => r.classes.reverse(),
]) {
  const value = structuredClone(result);
  mutate(value);
  assert.equal(sameEpisode(result, value), false);
}
for (const key of ["pool_hash", "study_id", "artifact_revision"]) {
  const missing = structuredClone(result);
  delete missing[key];
  assert.equal(sameEpisode(missing, structuredClone(missing)), false, key);
}
const legacy = structuredClone(result);
delete legacy.projection;
delete legacy.config;
assert.equal(
  validEpisode(legacy),
  true,
  "optional diagnostics remain optional",
);
const balanced = structuredClone(result);
balanced.decision_rule = "balanced_capacity";
for (const q of balanced.queries) {
  const i = (Number(q.predicted_id.slice(1)) + 1) % 5;
  q.predicted_id = `c${i}`;
  q.confidence = q.probabilities[i];
}
balanced.correct = 0;
balanced.accuracy = 0;
assert.equal(validEpisode(balanced), true);
balanced.queries[0].predicted_id = "c4";
balanced.queries[0].confidence = 0.05;
assert.equal(validEpisode(balanced), false);
console.log(
  "Lab04 result contracts passed: honest scores, balanced decisions, paired sample/revision identity and malformed input rejection.",
);

await import("./check-lab04-snapshot.mjs");

assert.equal(signedPoints(-1e-18), "0.00");
assert.equal(signedPoints(1e-18), "0.00");
assert.equal(signedPoints(0.0101), "+1.01");
assert.equal(signedPoints(-0.0101), "-1.01");
