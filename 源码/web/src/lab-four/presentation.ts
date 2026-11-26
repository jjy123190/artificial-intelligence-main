import type {
  Calibration,
  EpisodeRequest,
  EpisodeResult,
  Method,
} from "./types";

export const methodNames: Record<Method, string> = {
  baseline: "Baseline",
  maml: "MAML",
  calibrated: "特征校准",
};
export const calibrationNames: Record<Calibration, string> = {
  none: "仅训练对齐",
  mean: "均值校准",
  variance: "方差校准",
  prototype: "原型自适应",
  full: "完整特征校准",
  graph: "白化＋图传播校准",
  multiview: "双视图稳健校准",
  wide: "较宽骨干＋度量训练",
  wideview: "较宽骨干＋双视图校准",
  continued: "同容量续训＋双视图",
};
export const runKey = (run: Pick<EpisodeRequest, "method" | "calibration">) =>
  run.method === "calibrated" ? `calibrated:${run.calibration}` : run.method;
export const runName = (run: Pick<EpisodeRequest, "method" | "calibration">) =>
  run.method === "calibrated"
    ? calibrationNames[run.calibration]
    : methodNames[run.method];

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const isRecords = (value: unknown): value is Record<string, unknown>[] =>
  Array.isArray(value) && value.every(isRecord);
const isText = (value: unknown): value is string =>
  typeof value === "string" && value.trim().length > 0;
const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);
const isScore = (value: unknown): value is number =>
  isFiniteNumber(value) && value >= 0 && value <= 1;
const isPosition = (value: unknown): value is [number, number] =>
  Array.isArray(value) && value.length === 2 && value.every(isFiniteNumber);
const isSha256 = (value: unknown): value is string =>
  typeof value === "string" && /^[a-f0-9]{64}$/i.test(value);
const isChoice = (value: unknown, choices: Record<string, string>) =>
  typeof value === "string" &&
  Object.prototype.hasOwnProperty.call(choices, value);

// A numerical seed alone is not sufficient evidence of a paired comparison.
// Missing identities must never compare equal just because both are undefined.
export function sameEpisode(a: EpisodeResult, b: EpisodeResult): boolean {
  return (
    isSha256(a.pool_hash) &&
    isSha256(a.artifact_revision) &&
    isText(a.study_id) &&
    a.seed === b.seed &&
    a.pool_hash === b.pool_hash &&
    a.study_id === b.study_id &&
    a.artifact_revision === b.artifact_revision &&
    a.shots === b.shots &&
    a.way === b.way &&
    a.classes.length === b.classes.length &&
    a.classes.every((item, i) => item.id === b.classes[i].id) &&
    a.support.length === b.support.length &&
    a.queries.length === b.queries.length &&
    a.support.every(
      (item, i) =>
        item.id === b.support[i].id && item.class_id === b.support[i].class_id,
    ) &&
    a.queries.every(
      (item, i) =>
        item.id === b.queries[i].id &&
        item.target_id === b.queries[i].target_id,
    )
  );
}

// Treat API JSON as unknown until every field used by the result UI is checked.
// A rejected response leaves the previous verified result intact; never repair
// broken scores, invent missing points or let malformed data enter React/SVG.
export function validEpisode(value: unknown): value is EpisodeResult {
  if (
    !isRecord(value) ||
    value.compute !== "live" ||
    !isChoice(value.method, methodNames) ||
    !isChoice(value.calibration, calibrationNames) ||
    (value.method !== "calibrated" && value.calibration !== "none") ||
    !isScore(value.accuracy) ||
    value.way !== 5 ||
    (value.shots !== 1 && value.shots !== 5) ||
    !isRecords(value.classes) ||
    value.classes.length !== 5 ||
    !isRecords(value.support) ||
    value.support.length !== 5 * value.shots ||
    !isRecords(value.queries) ||
    value.total !== 75 ||
    value.queries.length !== value.total ||
    !isFiniteNumber(value.correct) ||
    !Number.isSafeInteger(value.correct) ||
    !isFiniteNumber(value.seed) ||
    !Number.isSafeInteger(value.seed) ||
    value.seed < 0 ||
    value.seed > 2_147_483_647 ||
    !isFiniteNumber(value.latency_ms) ||
    value.latency_ms < 0 ||
    !isSha256(value.checkpoint_sha256) ||
    !isSha256(value.pool_hash) ||
    !isSha256(value.artifact_revision) ||
    !isText(value.study_id) ||
    !isText(value.scope) ||
    !isText(value.metric) ||
    typeof value.transductive !== "boolean" ||
    (value.balanced_queries !== undefined &&
      typeof value.balanced_queries !== "boolean") ||
    (value.balanced_queries && !value.transductive) ||
    (value.decision_rule !== undefined &&
      value.decision_rule !== "score_argmax" &&
      value.decision_rule !== "balanced_capacity") ||
    (value.feature_motion !== "unchanged" &&
      value.feature_motion !== "real_support_adaptation") ||
    (value.prototype_role !== "descriptive_class_centers" &&
      value.prototype_role !== "decision_prototypes") ||
    (value.config !== undefined && !isRecord(value.config))
  )
    return false;

  const { classes, support, queries, points, prototypes, steps, projection } =
    value;
  const classIds = new Set(classes.map((c) => c.id));
  if (
    classIds.size !== 5 ||
    !classes.every((c) => isText(c.id) && isText(c.name)) ||
    !support.every(
      (s) => isText(s.id) && isText(s.image) && classIds.has(s.class_id),
    ) ||
    !queries.every(
      (q) =>
        isText(q.id) &&
        isText(q.image) &&
        classIds.has(q.target_id) &&
        classIds.has(q.predicted_id) &&
        isScore(q.confidence) &&
        Array.isArray(q.probabilities) &&
        q.probabilities.length === 5 &&
        q.probabilities.every(isScore),
    )
  )
    return false;

  const sampleClasses = new Map([
    ...support.map((s) => [s.id, s.class_id] as const),
    ...queries.map((q) => [q.id, q.target_id] as const),
  ]);
  const supportIds = new Set(support.map((s) => s.id));
  if (
    sampleClasses.size !== support.length + queries.length ||
    !classes.every(
      (c) =>
        support.filter((s) => s.class_id === c.id).length === value.shots &&
        queries.filter((q) => q.target_id === c.id).length === 15,
    ) ||
    (value.decision_rule === "balanced_capacity" &&
      (!value.balanced_queries ||
        !value.transductive ||
        !classes.every(
          (c) => queries.filter((q) => q.predicted_id === c.id).length === 15,
        ))) ||
    !isRecords(points) ||
    points.length !== sampleClasses.size ||
    new Set(points.map((p) => p.id)).size !== points.length ||
    !points.every(
      (p) =>
        sampleClasses.has(p.id) &&
        sampleClasses.get(p.id) === p.class_id &&
        p.kind === (supportIds.has(p.id) ? "support" : "query") &&
        isPosition(p.before) &&
        isPosition(p.after),
    ) ||
    !isRecords(prototypes) ||
    prototypes.length !== classes.length ||
    new Set(prototypes.map((p) => p.class_id)).size !== classes.length ||
    !prototypes.every(
      (p) =>
        classIds.has(p.class_id) && isPosition(p.before) && isPosition(p.after),
    ) ||
    !isRecords(steps) ||
    steps.length === 0 ||
    !steps.every((s) => isText(s.name) && isText(s.detail))
  )
    return false;

  if (
    projection !== undefined &&
    (!isRecord(projection) ||
      projection.method !== "PCA" ||
      !Array.isArray(projection.explained_variance) ||
      projection.explained_variance.length !== 2 ||
      !projection.explained_variance.every(isScore) ||
      projection.explained_variance.reduce((sum, n) => sum + n, 0) > 1 + 1e-6 ||
      !isFiniteNumber(projection.feature_dimensions) ||
      !Number.isSafeInteger(projection.feature_dimensions) ||
      projection.feature_dimensions < 2 ||
      projection.computed_after_prediction !== true)
  )
    return false;

  return (
    queries.every((q) => {
      // Element types have been checked above; no numeric coercion is allowed.
      const scores = q.probabilities as number[];
      const predictedIndex = classes.findIndex((c) => c.id === q.predicted_id);
      return (
        Math.abs(scores.reduce((a, b) => a + b, 0) - 1) < 1e-5 &&
        Math.abs((q.confidence as number) - scores[predictedIndex]) < 1e-5 &&
        (value.decision_rule === "balanced_capacity" ||
          scores.indexOf(Math.max(...scores)) === predictedIndex)
      );
    }) &&
    value.correct ===
      queries.filter((q) => q.predicted_id === q.target_id).length &&
    Math.abs(value.correct / value.total - value.accuracy) < 1e-6
  );
}
export function signedPoints(value: number, digits = 2) {
  const rounded = Number((value * 100).toFixed(digits));
  return `${rounded > 0 ? "+" : ""}${(Object.is(rounded, -0) ? 0 : rounded).toFixed(digits)}`;
}
