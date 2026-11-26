/** Unknown JSON and statistical consistency; fixtures are not measured results. */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";
const source = await readFile(
  new URL("../src/lab-four/snapshot-contract.ts", import.meta.url),
  "utf8",
);
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
});
const { validSnapshot } = await import(
  "data:text/javascript;base64," +
    Buffer.from(
      outputText + "\n//# sourceURL=lab04-snapshot-contract.js",
    ).toString("base64")
);
const empty = {
  available: false,
  status: "Not ready",
  title: "Fixture",
  dataset: {
    name: "mini-ImageNet",
    image_size: 84,
    train_classes: 64,
    val_classes: 16,
    test_classes: 20,
    training_images_per_class: 600,
    protocol_note: "Fixture",
  },
  model: { name: "ResNet12", channels: [8, 16, 32, 64], parameters: 130160 },
  methods: ["baseline", "maml", "calibrated"].map((id) => ({
    id,
    name: id,
    description: "Fixture",
  })),
  comparisons: [],
  ablations: [],
  training: [],
  limitations: [],
  provenance: { seed: 20261124, checkpoint_sha256: "" },
};
const measured = structuredClone(empty);
Object.assign(measured, {
  available: true,
  provenance: {
    seed: 20261124,
    checkpoint_sha256: "a".repeat(64),
    artifact_revision: "b".repeat(64),
    run_id: "2026-11-25",
  },
  verification: {
    verified: true,
    core_tasks: 1200,
    extension_tasks: 4800,
    scope: "Fixture",
  },
});
const methods = {
  baseline: 0.4,
  maml: 0.3,
  full: 0.5,
  graph: 0.55,
  multiview: 0.56,
  wide: 0.57,
  wideview: 0.58,
};
measured.comparisons = [1, 5].flatMap((shot) =>
  Object.entries(methods).map(([method, accuracy]) => ({
    method,
    label: method,
    shot,
    accuracy,
    ci95: 0.01,
    episodes: 600,
    timing_scope: "fixture",
    paired_delta: accuracy - 0.4,
    paired_ci95: 0.01,
    transductive: method !== "baseline" && method !== "maml",
    balanced_queries: method !== "baseline" && method !== "maml",
  })),
);
measured.ablations = measured.comparisons
  .filter((r) => r.method !== "baseline" && r.method !== "maml")
  .map((r) => ({ ...r }));
measured.training = [
  {
    epoch: 1,
    loss: 2,
    accuracy: 0.5,
    validation_1shot: 0.4,
    validation_5shot: 0.6,
  },
];
measured.limitations = ["Fixture only"];
for (const name of ["graph", "multiview", "wideview"])
  measured[name + "_extension"] = { available: true, description: "Fixture" };
measured.wide_extension = {
  available: true,
  description: "Fixture",
  model: { name: "Wide", widths: [16, 32, 64, 128], parameters: 509856 },
  training: { selected_epoch: 24, epochs_completed: 24, warm_start: "Fixture" },
};
measured.calibration = Object.fromEntries(
  [1, 5].map((n) => [
    String(n),
    {
      mean_weight: 0.1,
      variance_blend: 0.2,
      balanced_queries: true,
      prototype_steps: 4,
      prototype_weight: 0.2,
      temperature: 0.1,
    },
  ]),
);
measured.study = {
  version: "v2",
  name: "Fixture",
  selection: "validation",
  limitations: ["Fixture"],
  feature_spaces: { 1: { center_strength: 1, power: 1 } },
};
measured.history = [
  {
    version: "v1",
    label: "Fixture",
    model: { name: "Narrow", embedding_dim: 32 },
    protocol: {},
    rows: structuredClone(
      measured.comparisons.filter((r) =>
        ["baseline", "full", "maml"].includes(r.method),
      ),
    ),
  },
];
measured.paired_history = [];
measured.diagnostics = {
  method: "multiview",
  reference: "graph",
  scope: "Fixture",
  rows: [1, 5].map((shots) => ({
    shots,
    episodes: 600,
    query_occurrences: 45000,
    task_p05: 0.3,
    task_median: 0.56,
    task_p95: 0.8,
    paired_wins: 200,
    paired_ties: 150,
    paired_losses: 250,
    ece10: 0.01,
    brier_score: 0.4,
    negative_log_likelihood: 1,
    score_bins: Array.from({ length: 10 }, (_, i) => ({
      lower: i / 10,
      upper: (i + 1) / 10,
      occurrences: i === 5 ? 45000 : 0,
      mean_score: i === 5 ? 0.55 : null,
      accuracy: i === 5 ? 0.56 : null,
    })),
  })),
};
measured.edge_benchmark = {
  device: "CPU",
  scope: "Fixture",
  rows: measured.comparisons.map((r) => ({
    method: r.method === "full" ? "calibrated" : r.method,
    shots: r.shot,
    repetitions: 20,
    encoding_and_classification_median_ms: 100,
    encoding_and_classification_p95_ms: 110,
    warm_wall_median_ms: 130,
  })),
};
assert.equal(validSnapshot(empty), true, "honest empty snapshot");
assert.equal(validSnapshot(measured), true, "full snapshot");
assert.equal(
  validSnapshot({ ...measured, available: false }),
  true,
  "missing runtime does not discard measured evidence",
);
const unavailableWide = structuredClone(empty);
unavailableWide.wide_extension = {
  available: false,
  description: "Missing",
  model: null,
  training: null,
};
assert.equal(validSnapshot(unavailableWide), true);
for (const value of [
  null,
  undefined,
  true,
  7,
  "fixture",
  [],
  {},
  { ...empty, model: null },
])
  assert.equal(validSnapshot(value), false);
measured.module_contrasts = {
  scope: "Frozen task fixture",
  interval_note: "Pointwise intervals, no multiple-test adjustment",
  rows: [1, 5].map((shot) => ({
    id: "example",
    label: "Fixture comparison",
    method: "full",
    reference: "baseline",
    scope: "Fixture only",
    shot,
    episodes: 600,
    method_accuracy: 0.5,
    reference_accuracy: 0.4,
    paired_delta: 0.1,
    paired_ci95: 0.02,
    ci95_low: 0.08,
    ci95_high: 0.12,
    wins: 300,
    ties: 150,
    losses: 150,
  })),
};
assert.equal(validSnapshot(measured), true, "valid module pair statistics");
const continued = structuredClone(measured);
continued.continued_extension = {
  available: true,
  description: "Fixed-capacity training fixture",
  model: { name: "Continued", widths: [16, 32, 64, 128], parameters: 509856 },
  training: {
    additional_epochs: 16,
    selected_additional_epoch: 12,
    original_wide_epochs: 24,
    same_capacity: true,
  },
};
for (const shot of [1, 5])
  continued.comparisons.push({
    method: "continued",
    label: "Continued",
    shot,
    accuracy: 0.6,
    ci95: 0.01,
    episodes: 600,
    timing_scope: "fixture",
    paired_delta: 0.2,
    paired_ci95: 0.01,
    paired_vs_wideview: 0.02,
    paired_vs_wideview_ci95: 0.01,
  });
continued.verification.extension_tasks = 6000;
assert.equal(
  validSnapshot(continued),
  true,
  "continued same-capacity extension",
);
for (const mutate of [
  (v) => v.continued_extension.model.parameters++,
  (v) => (v.continued_extension.training.selected_additional_epoch = 30),
  (v) => (v.continued_extension.training.additional_epochs = 0),
  (v) => (v.continued_extension.training.same_capacity = false),
  (v) => (v.continued_extension.model = null),
  (v) => (v.comparisons.at(-1).paired_vs_wideview = 0.5),
]) {
  const invalid = structuredClone(continued);
  mutate(invalid);
  assert.equal(
    validSnapshot(invalid),
    false,
    "invalid continuation identity or budget",
  );
}
const mutations = [
  (v) => (v.module_contrasts.rows[0] = null),
  (v) => (v.module_contrasts.rows[0].paired_delta = 0.9),
  (v) => (v.module_contrasts.rows[0].paired_ci95 = -0.1),
  (v) => (v.module_contrasts.rows[0].ci95_low = 0.9),
  (v) => (v.module_contrasts.rows[0].wins = 999),
  (v) => (v.module_contrasts.rows[0].reference = "unknown"),
  (v) => (v.module_contrasts.rows[0].scope = {}),
  (v) => v.module_contrasts.rows.push(v.module_contrasts.rows[0]),
  (v) => (v.module_contrasts.interval_note = null),

  (v) => (v.methods[0].id = ["baseline"]),
  (v) => (v.comparisons[0].decision_rule = ["score_argmax"]),
  (v) => (v.edge_benchmark.rows[0].method = ["baseline"]),
  (v) => (v.model.channels = null),
  (v) => (v.model.channels = [1, 2]),
  (v) => (v.model.channels[0] = NaN),
  (v) => (v.model.parameters = {}),
  (v) => (v.dataset = null),
  (v) => (v.dataset.protocol_note = {}),
  (v) => (v.dataset.image_size = 0),
  (v) => (v.methods[0] = null),
  (v) => (v.methods[0].name = {}),
  (v) => (v.methods[0].id = "unknown"),
  (v) => v.methods.pop(),
  (v) => (v.comparisons = null),
  (v) => (v.comparisons[0] = null),
  (v) => (v.comparisons[0].accuracy = "0.4"),
  (v) => (v.comparisons[0].ci95 = -1),
  (v) => (v.comparisons[0].accuracy = 0.9),
  (v) => (v.comparisons[1].paired_delta = 0.7),
  (v) => delete v.comparisons[1].paired_ci95,
  (v) => (v.comparisons[0].episodes = 601),
  (v) => v.comparisons.push(v.comparisons[0]),
  (v) => (v.comparisons = v.comparisons.filter((r) => r.method !== "baseline")),
  (v) => (v.ablations[0].accuracy = 0.1),
  (v) => (v.ablations[0].label = {}),
  (v) => (v.ablations = [null]),
  (v) => (v.training[0] = null),
  (v) => (v.training[0].loss = NaN),
  (v) => (v.training[0].validation_5shot = 2),
  (v) => v.training.push(v.training[0]),
  (v) => (v.limitations = [{}]),
  (v) => (v.provenance = null),
  (v) => (v.provenance.checkpoint_sha256 = ""),
  (v) => delete v.provenance.artifact_revision,
  (v) => (v.verification.verified = false),
  (v) => (v.verification.core_tasks = 0),
  (v) => (v.verification.extension_tasks = 1200),
  (v) => (v.graph_extension.available = "true"),
  (v) => (v.graph_extension.available = false),
  (v) => (v.wide_extension.model = null),
  (v) => (v.wide_extension.training.selected_epoch = 99),
  (v) => (v.wide_extension.model.widths = null),
  (v) => (v.wideview_extension.description = {}),
  (v) => delete v.wide_extension,
  (v) => (v.calibration["1"].prototype_steps = -1),
  (v) => (v.calibration["5"].temperature = 0),
  (v) => (v.calibration["1"] = null),
  (v) => (v.study.limitations = [null]),
  (v) => (v.study.feature_spaces["1"].center_strength = NaN),
  (v) => (v.study.name = {}),
  (v) => (v.history[0] = null),
  (v) => (v.history[0].model.embedding_dim = 0),
  (v) => (v.history[0].rows[0].accuracy = Infinity),
  (v) => (v.history[0].label = {}),
  (v) => (v.diagnostics.method = "wideview"),
  (v) => (v.diagnostics.rows[0] = null),
  (v) => (v.diagnostics.rows[0].score_bins = null),
  (v) => (v.diagnostics.rows[0].score_bins[0] = null),
  (v) => (v.diagnostics.rows[0].score_bins[0].mean_score = 0.2),
  (v) => (v.diagnostics.rows[0].score_bins[5].occurrences = 12),
  (v) => (v.diagnostics.rows[0].score_bins[5].mean_score = 0.7),
  (v) => (v.diagnostics.rows[0].score_bins[5].accuracy = 0.1),
  (v) => (v.diagnostics.rows[0].ece10 = 0.7),
  (v) => (v.diagnostics.rows[0].query_occurrences = 300),
  (v) => (v.diagnostics.rows[0].paired_wins = 1),
  (v) => (v.diagnostics.rows[0].task_p05 = 0.99),
  (v) => (v.diagnostics.rows[0].negative_log_likelihood = -1),
  (v) => (v.edge_benchmark.rows[0] = null),
  (v) => (v.edge_benchmark.rows[0].encoding_and_classification_median_ms = 200),
  (v) => (v.edge_benchmark.rows[0].repetitions = 1),
  (v) => (v.edge_benchmark.rows[0].method = "unknown"),
];
for (const [i, mutate] of mutations.entries()) {
  const value = structuredClone(measured);
  mutate(value);
  assert.equal(validSnapshot(value), false, "rejected mutation " + i);
}
console.log(
  `Lab04 snapshot contract: valid empty/unavailable/current snapshots, ${mutations.length} nested/statistical mutations rejected.`,
);
