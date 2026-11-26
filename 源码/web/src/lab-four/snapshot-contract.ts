import type { Comparison, LabFourSnapshot } from "./types";

const record = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === "object" && !Array.isArray(v);
const records = (v: unknown): v is Record<string, unknown>[] =>
  Array.isArray(v) && v.every(record);
const text = (v: unknown): v is string =>
  typeof v === "string" && v.trim().length > 0;
const texts = (v: unknown): v is string[] => Array.isArray(v) && v.every(text);
const finite = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);
const nonnegative = (v: unknown): v is number => finite(v) && v >= 0;
const integer = (v: unknown): v is number =>
  nonnegative(v) && Number.isSafeInteger(v);
const positiveInteger = (v: unknown): v is number => integer(v) && v > 0;
const score = (v: unknown): v is number => finite(v) && v >= 0 && v <= 1;
const delta = (v: unknown): v is number => finite(v) && v >= -1 && v <= 1;
const boolean = (v: unknown): v is boolean => typeof v === "boolean";
const hash = (v: unknown): v is string =>
  typeof v === "string" && /^[a-f0-9]{64}$/i.test(v);
const widths = (v: unknown) =>
  Array.isArray(v) && v.length === 4 && v.every(positiveInteger);
const close = (a: number, b: number) => Math.abs(a - b) <= 1e-7;
const optional = (v: unknown, test: (v: unknown) => boolean) =>
  v === undefined || test(v);
const optionalNullable = (v: unknown, test: (v: unknown) => boolean) =>
  v == null || test(v);
const unique = (values: unknown[]) => new Set(values).size === values.length;
const pairFields = [
  ["paired_delta", "paired_ci95"],
  ["paired_vs_full", "paired_vs_full_ci95"],
  ["paired_vs_graph", "paired_vs_graph_ci95"],
  ["paired_vs_multiview", "paired_vs_multiview_ci95"],
  ["paired_vs_wide", "paired_vs_wide_ci95"],
  ["paired_vs_wideview", "paired_vs_wideview_ci95"],
] as const;

function comparisons(v: unknown): v is Comparison[] {
  return (
    records(v) &&
    unique(v.map((r) => `${r.method}:${r.shot}`)) &&
    v.every(
      (r) =>
        text(r.method) &&
        text(r.label) &&
        (r.shot === 1 || r.shot === 5) &&
        score(r.accuracy) &&
        nonnegative(r.ci95) &&
        r.ci95 <= 1 &&
        positiveInteger(r.episodes) &&
        text(r.timing_scope) &&
        optional(r.latency_ms, nonnegative) &&
        optional(r.transductive, boolean) &&
        optional(r.balanced_queries, boolean) &&
        (!r.balanced_queries || r.transductive === true) &&
        optional(r.raw_accuracy, score) &&
        optional(
          r.decision_rule,
          (rule) =>
            ["score_argmax", "soft", "balanced_capacity"].includes(
              rule as string,
            ) && typeof rule === "string",
        ) &&
        pairFields.every(
          ([mean, ci]) =>
            (r[mean] == null && r[ci] == null) ||
            (delta(r[mean]) && nonnegative(r[ci]) && r[ci] <= 2),
        ),
    )
  );
}

function diagnostics(v: unknown): boolean {
  if (
    !record(v) ||
    v.method !== "multiview" ||
    v.reference !== "graph" ||
    !text(v.scope) ||
    !records(v.rows) ||
    !unique(v.rows.map((r) => r.shots))
  )
    return false;
  return v.rows.every((r) => {
    if (
      (r.shots !== 1 && r.shots !== 5) ||
      !positiveInteger(r.episodes) ||
      !positiveInteger(r.query_occurrences) ||
      r.query_occurrences !== r.episodes * 75 ||
      !score(r.task_p05) ||
      !score(r.task_median) ||
      !score(r.task_p95) ||
      r.task_p05 > r.task_median ||
      r.task_median > r.task_p95 ||
      !integer(r.paired_wins) ||
      !integer(r.paired_ties) ||
      !integer(r.paired_losses) ||
      r.paired_wins + r.paired_ties + r.paired_losses !== r.episodes ||
      !score(r.ece10) ||
      !nonnegative(r.brier_score) ||
      r.brier_score > 2 ||
      !nonnegative(r.negative_log_likelihood) ||
      !records(r.score_bins) ||
      r.score_bins.length !== 10
    )
      return false;
    let count = 0,
      ece = 0;
    for (const [i, bin] of r.score_bins.entries()) {
      if (
        !score(bin.lower) ||
        !score(bin.upper) ||
        !close(bin.lower, i / 10) ||
        !close(bin.upper, (i + 1) / 10) ||
        !integer(bin.occurrences)
      )
        return false;
      if (bin.occurrences === 0) {
        if (bin.mean_score !== null || bin.accuracy !== null) return false;
      } else {
        if (
          !score(bin.mean_score) ||
          !score(bin.accuracy) ||
          bin.mean_score < bin.lower - 1e-8 ||
          bin.mean_score > bin.upper + 1e-8
        )
          return false;
        ece += bin.occurrences * Math.abs(bin.mean_score - bin.accuracy);
      }
      count += bin.occurrences;
    }
    return count === r.query_occurrences && close(ece / count, r.ece10);
  });
}

function timing(v: unknown): boolean {
  return (
    record(v) &&
    text(v.device) &&
    text(v.scope) &&
    records(v.rows) &&
    unique(v.rows.map((r) => `${r.method}:${r.shots}`)) &&
    v.rows.every(
      (r) =>
        [
          "baseline",
          "maml",
          "calibrated",
          "graph",
          "multiview",
          "wide",
          "wideview",
          "continued",
        ].includes(r.method as string) &&
        typeof r.method === "string" &&
        (r.shots === 1 || r.shots === 5) &&
        integer(r.repetitions) &&
        r.repetitions >= 3 &&
        nonnegative(r.encoding_and_classification_median_ms) &&
        nonnegative(r.encoding_and_classification_p95_ms) &&
        r.encoding_and_classification_median_ms <=
          r.encoding_and_classification_p95_ms &&
        nonnegative(r.warm_wall_median_ms),
    )
  );
}

function moduleContrasts(v: unknown): boolean {
  return (
    record(v) &&
    text(v.scope) &&
    text(v.interval_note) &&
    records(v.rows) &&
    unique(v.rows.map((r) => `${r.id}:${r.shot}`)) &&
    v.rows.every(
      (r) =>
        ["id", "label", "method", "reference", "scope"].every((key) =>
          text(r[key]),
        ) &&
        r.method !== r.reference &&
        (r.shot === 1 || r.shot === 5) &&
        positiveInteger(r.episodes) &&
        score(r.method_accuracy) &&
        score(r.reference_accuracy) &&
        delta(r.paired_delta) &&
        nonnegative(r.paired_ci95) &&
        r.paired_ci95 <= 2 &&
        finite(r.ci95_low) &&
        finite(r.ci95_high) &&
        close(r.paired_delta, r.method_accuracy - r.reference_accuracy) &&
        close(r.ci95_low, r.paired_delta - r.paired_ci95) &&
        close(r.ci95_high, r.paired_delta + r.paired_ci95) &&
        integer(r.wins) &&
        integer(r.ties) &&
        integer(r.losses) &&
        r.wins + r.ties + r.losses === r.episodes,
    )
  );
}

function calibration(v: unknown): boolean {
  return (
    record(v) &&
    Object.entries(v).every(
      ([key, row]) =>
        ["1", "5"].includes(key) &&
        record(row) &&
        score(row.mean_weight) &&
        optional(row.variance_blend, score) &&
        optional(row.balanced_queries, boolean) &&
        integer(row.prototype_steps) &&
        row.prototype_steps <= 50 &&
        nonnegative(row.prototype_weight) &&
        finite(row.temperature) &&
        row.temperature > 0,
    )
  );
}

function study(v: unknown): boolean {
  return (
    record(v) &&
    ["version", "name", "selection"].every((key) => optional(v[key], text)) &&
    optional(v.limitations, texts) &&
    optional(
      v.feature_spaces,
      (spaces) =>
        record(spaces) &&
        Object.entries(spaces).every(
          ([key, row]) =>
            ["1", "5"].includes(key) &&
            record(row) &&
            score(row.center_strength) &&
            finite(row.power) &&
            row.power > 0,
        ),
    )
  );
}

function wide(v: unknown): boolean {
  if (!record(v) || !boolean(v.available) || !text(v.description)) return false;
  const model = v.model,
    train = v.training;
  if (
    !(
      model === null ||
      (record(model) &&
        text(model.name) &&
        widths(model.widths) &&
        positiveInteger(model.parameters))
    )
  )
    return false;
  if (
    !(
      train === null ||
      (record(train) &&
        integer(train.selected_epoch) &&
        positiveInteger(train.epochs_completed) &&
        train.selected_epoch <= train.epochs_completed &&
        text(train.warm_start))
    )
  )
    return false;
  return !v.available || (model !== null && train !== null);
}

/** Validate both the empty/unavailable snapshot and every field consumed by the
 * report pages. It checks internal consistency, not authenticity: server-side
 * artifact recomputation remains the source of verified experimental evidence. */
export function validSnapshot(value: unknown): value is LabFourSnapshot {
  if (
    !record(value) ||
    !boolean(value.available) ||
    !text(value.status) ||
    !text(value.title) ||
    !record(value.dataset) ||
    !record(value.model) ||
    !records(value.methods) ||
    !comparisons(value.comparisons) ||
    !comparisons(value.ablations) ||
    !records(value.training) ||
    !texts(value.limitations) ||
    !record(value.provenance)
  )
    return false;
  const { dataset, model, methods, training, provenance } = value;
  if (
    !text(dataset.name) ||
    !text(dataset.protocol_note) ||
    ![
      "image_size",
      "train_classes",
      "val_classes",
      "test_classes",
      "training_images_per_class",
    ].every((key) => positiveInteger(dataset[key])) ||
    !text(model.name) ||
    !widths(model.channels) ||
    !positiveInteger(model.parameters) ||
    !optional(model.alignment_weight, nonnegative) ||
    methods.length !== 3 ||
    !unique(methods.map((m) => m.id)) ||
    !methods.every(
      (m) =>
        ["baseline", "maml", "calibrated"].includes(m.id as string) &&
        typeof m.id === "string" &&
        text(m.name) &&
        text(m.description),
    ) ||
    !unique(training.map((r) => r.epoch)) ||
    !training.every(
      (r) =>
        positiveInteger(r.epoch) &&
        nonnegative(r.loss) &&
        optional(r.accuracy, score) &&
        optionalNullable(r.validation_1shot, score) &&
        optionalNullable(r.validation_5shot, score),
    ) ||
    !integer(provenance.seed) ||
    provenance.seed > 4294967295 ||
    !(
      hash(provenance.checkpoint_sha256) ||
      (!value.available &&
        value.comparisons.length === 0 &&
        provenance.checkpoint_sha256 === "")
    ) ||
    !optional(provenance.artifact_revision, hash) ||
    !optional(provenance.run_id, text)
  )
    return false;
  if (
    !optional(
      value.verification,
      (v) =>
        record(v) &&
        boolean(v.verified) &&
        integer(v.core_tasks) &&
        integer(v.extension_tasks) &&
        text(v.scope),
    ) ||
    !optional(value.paired_history, comparisons) ||
    !optional(value.module_contrasts, moduleContrasts) ||
    !optional(value.calibration, calibration) ||
    !optional(value.study, study) ||
    !optionalNullable(value.edge_benchmark, timing) ||
    !optionalNullable(value.diagnostics, diagnostics) ||
    !optional(value.wide_extension, wide) ||
    !optional(
      value.continued_extension,
      (v) =>
        record(v) &&
        boolean(v.available) &&
        text(v.description) &&
        (v.model === null ||
          (record(v.model) &&
            text(v.model.name) &&
            widths(v.model.widths) &&
            positiveInteger(v.model.parameters))) &&
        (v.training === null ||
          (record(v.training) &&
            positiveInteger(v.training.additional_epochs) &&
            positiveInteger(v.training.original_wide_epochs) &&
            integer(v.training.selected_additional_epoch) &&
            v.training.selected_additional_epoch <=
              v.training.additional_epochs &&
            v.training.same_capacity === true)) &&
        (!v.available || (v.model !== null && v.training !== null)),
    ) ||
    !["graph_extension", "multiview_extension", "wideview_extension"].every(
      (key) =>
        optional(
          value[key],
          (v) => record(v) && boolean(v.available) && text(v.description),
        ),
    ) ||
    !optional(
      value.history,
      (v) =>
        records(v) &&
        unique(v.map((h) => h.version)) &&
        v.every(
          (h) =>
            text(h.version) &&
            text(h.label) &&
            record(h.model) &&
            text(h.model.name) &&
            positiveInteger(h.model.embedding_dim) &&
            record(h.protocol) &&
            comparisons(h.rows),
        ),
    )
  )
    return false;

  // Nested shapes are checked above; now reject contradictory but well-typed data.
  const snapshot = value as LabFourSnapshot;
  const rows = snapshot.comparisons;
  if (snapshot.available && !rows.length) return false;
  if (
    rows.length &&
    (!hash(provenance.artifact_revision) ||
      snapshot.verification?.verified !== true)
  )
    return false;
  for (const shot of [1, 5]) {
    const selected = rows.filter((r) => r.shot === shot),
      baseline = selected.find((r) => r.method === "baseline");
    if (
      rows.length &&
      (!baseline ||
        !selected.some((r) => r.method === "maml") ||
        !selected.some((r) => ["full", "calibrated"].includes(r.method)))
    )
      return false;
    if (baseline && selected.some((r) => r.episodes !== baseline.episodes))
      return false;
  }
  const all = [
    ...rows,
    ...snapshot.ablations,
    ...(snapshot.paired_history || []),
  ];
  for (const row of all) {
    for (const [field, reference] of [
      ["paired_delta", "baseline"],
      ["paired_vs_full", "full"],
      ["paired_vs_graph", "graph"],
      ["paired_vs_multiview", "multiview"],
      ["paired_vs_wide", "wide"],
      ["paired_vs_wideview", "wideview"],
    ] as const) {
      const mean = row[field];
      if (mean == null) continue;
      const ref = rows.find(
        (r) =>
          r.shot === row.shot &&
          (r.method === reference ||
            (reference === "full" && r.method === "calibrated")),
      );
      if (
        !ref ||
        row.episodes !== ref.episodes ||
        !close(mean, row.accuracy - ref.accuracy)
      )
        return false;
    }
    const repeated = rows.find(
      (r) => r.method === row.method && r.shot === row.shot,
    );
    if (
      repeated &&
      (!close(repeated.accuracy, row.accuracy) ||
        !close(repeated.ci95, row.ci95) ||
        repeated.episodes !== row.episodes)
    )
      return false;
  }
  for (const contrast of snapshot.module_contrasts?.rows || []) {
    const selected = all.find(
      (r) => r.method === contrast.method && r.shot === contrast.shot,
    );
    const reference = all.find(
      (r) => r.method === contrast.reference && r.shot === contrast.shot,
    );
    if (
      !selected ||
      !reference ||
      selected.episodes !== contrast.episodes ||
      reference.episodes !== contrast.episodes ||
      !close(selected.accuracy, contrast.method_accuracy) ||
      !close(reference.accuracy, contrast.reference_accuracy)
    )
      return false;
  }
  let extensionTasks = 0;
  for (const method of [
    "graph",
    "multiview",
    "wide",
    "wideview",
    "continued",
  ] as const) {
    const available = snapshot[`${method}_extension`]?.available === true;
    const ext = rows.filter((r) => r.method === method);
    if (available ? ext.length !== 2 : ext.length !== 0) return false;
    extensionTasks += ext.reduce((total, r) => total + r.episodes, 0);
  }
  if (
    snapshot.continued_extension?.available &&
    (!snapshot.wideview_extension?.available ||
      snapshot.continued_extension.model?.parameters !==
        snapshot.wide_extension?.model?.parameters)
  )
    return false;
  if (
    snapshot.wideview_extension?.available &&
    !snapshot.wide_extension?.available
  )
    return false;
  if (
    snapshot.verification?.verified &&
    (snapshot.verification.core_tasks !==
      rows
        .filter((r) => r.method === "baseline")
        .reduce((n, r) => n + r.episodes, 0) ||
      snapshot.verification.extension_tasks !== extensionTasks)
  )
    return false;
  if (
    snapshot.edge_benchmark?.rows.some(
      (r) =>
        !rows.some(
          (c) =>
            c.shot === r.shots &&
            c.method === (r.method === "calibrated" ? "full" : r.method),
        ),
    )
  )
    return false;
  for (const diagnostic of snapshot.diagnostics?.rows || []) {
    const source = rows.find(
      (r) =>
        r.method === snapshot.diagnostics!.method &&
        r.shot === diagnostic.shots,
    );
    const reference = rows.find(
      (r) =>
        r.method === snapshot.diagnostics!.reference &&
        r.shot === diagnostic.shots,
    );
    const accuracy =
      diagnostic.score_bins.reduce(
        (total, bin) => total + (bin.accuracy || 0) * bin.occurrences,
        0,
      ) / diagnostic.query_occurrences;
    if (
      !source ||
      !reference ||
      source.episodes !== diagnostic.episodes ||
      !close(source.accuracy, accuracy)
    )
      return false;
  }
  return true;
}
