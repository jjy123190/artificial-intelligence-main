export type Method = "baseline" | "maml" | "calibrated";
export type Calibration =
  | "none"
  | "mean"
  | "variance"
  | "prototype"
  | "full"
  | "graph"
  | "multiview"
  | "wide"
  | "wideview"
  | "continued";
export type EpisodeRequest = {
  method: Method;
  shots: 1 | 5;
  seed: number;
  calibration: Calibration;
};
export type Comparison = {
  method: string;
  label: string;
  shot: number;
  accuracy: number;
  ci95: number;
  episodes: number;
  latency_ms?: number;
  timing_scope: string;
  paired_delta?: number | null;
  paired_ci95?: number | null;
  paired_vs_full?: number;
  paired_vs_full_ci95?: number;
  paired_vs_graph?: number;
  paired_vs_graph_ci95?: number;
  paired_vs_multiview?: number;
  paired_vs_multiview_ci95?: number;
  paired_vs_wide?: number;
  paired_vs_wide_ci95?: number;
  paired_vs_wideview?: number;
  paired_vs_wideview_ci95?: number;
  raw_accuracy?: number;
  decision_rule?: "score_argmax" | "soft" | "balanced_capacity";
  transductive?: boolean;
  balanced_queries?: boolean;
};
export type ModuleContrast = {
  id: string;
  label: string;
  method: string;
  reference: string;
  scope: string;
  shot: 1 | 5;
  episodes: number;
  method_accuracy: number;
  reference_accuracy: number;
  paired_delta: number;
  paired_ci95: number;
  ci95_low: number;
  ci95_high: number;
  wins: number;
  ties: number;
  losses: number;
};
export type LabFourSnapshot = {
  continued_extension?: {
    available: boolean;
    description: string;
    model: { name: string; widths: number[]; parameters: number } | null;
    training: {
      additional_epochs: number;
      selected_additional_epoch: number;
      original_wide_epochs: number;
      same_capacity: boolean;
    } | null;
  };
  module_contrasts?: {
    rows: ModuleContrast[];
    scope: string;
    interval_note: string;
  };
  wideview_extension?: { available: boolean; description: string };
  wide_extension?: {
    available: boolean;
    model: { name: string; widths: number[]; parameters: number } | null;
    training: {
      selected_epoch: number;
      epochs_completed: number;
      warm_start: string;
    } | null;
    description: string;
  };
  diagnostics?: {
    method: string;
    reference: string;
    scope: string;
    rows: {
      shots: number;
      episodes: number;
      query_occurrences: number;
      task_p05: number;
      task_median: number;
      task_p95: number;
      paired_wins: number;
      paired_ties: number;
      paired_losses: number;
      ece10: number;
      brier_score: number;
      negative_log_likelihood: number;
      score_bins: {
        lower: number;
        upper: number;
        occurrences: number;
        mean_score: number | null;
        accuracy: number | null;
      }[];
    }[];
  } | null;
  verification?: {
    verified: boolean;
    core_tasks: number;
    extension_tasks: number;
    scope: string;
  };
  multiview_extension?: { available: boolean; description: string };
  graph_extension?: {
    available: boolean;
    description: string;
  };
  available: boolean;
  status: string;
  title: string;
  dataset: {
    name: string;
    image_size: number;
    train_classes: number;
    val_classes: number;
    test_classes: number;
    training_images_per_class: number;
    protocol_note: string;
  };
  model: {
    name: string;
    channels: number[];
    parameters: number;
    alignment_weight?: number;
  };
  methods: { id: Method; name: string; description: string }[];
  comparisons: Comparison[];
  ablations: Comparison[];
  training: {
    epoch: number;
    loss: number;
    accuracy?: number;
    validation_1shot?: number | null;
    validation_5shot?: number | null;
  }[];
  history?: {
    version: string;
    label: string;
    model: { name: string; embedding_dim: number };
    protocol: Record<string, unknown>;
    rows: Comparison[];
  }[];
  paired_history?: Comparison[];
  calibration?: Record<
    string,
    {
      mean_weight: number;
      variance_blend?: number;
      balanced_queries?: boolean;
      prototype_steps: number;
      prototype_weight: number;
      temperature: number;
    }
  >;
  study?: {
    version?: string;
    name?: string;
    selection?: string;
    limitations?: string[];
    feature_spaces?: Record<string, { center_strength: number; power: number }>;
  };
  edge_benchmark?: {
    device: string;
    scope: string;
    rows: {
      method: string;
      shots: number;
      repetitions: number;
      encoding_and_classification_median_ms: number;
      encoding_and_classification_p95_ms: number;
      warm_wall_median_ms: number;
    }[];
  } | null;
  limitations: string[];
  provenance: {
    seed: number;
    checkpoint_sha256: string;
    run_id?: string;
    artifact_revision?: string;
  };
};
export type Position = [number, number];
export type EpisodeResult = {
  method: Method;
  shots: 1 | 5;
  seed: number;
  way: number;
  accuracy: number;
  correct: number;
  total: number;
  latency_ms: number;
  classes: { id: string; name: string }[];
  support: { id: string; class_id: string; image: string }[];
  queries: {
    id: string;
    image: string;
    predicted_id: string;
    target_id: string;
    confidence: number;
    probabilities: number[];
  }[];
  prototypes: { class_id: string; before: Position; after: Position }[];
  points: {
    id: string;
    class_id: string;
    kind: "support" | "query";
    before: Position;
    after: Position;
  }[];
  steps: { name: string; detail: string }[];
  scope: string;
  checkpoint_sha256: string;
  compute: "live";
  transductive: boolean;
  metric: string;
  decision_rule?: "score_argmax" | "balanced_capacity";
  projection?: {
    method: "PCA";
    explained_variance: number[];
    feature_dimensions: number;
    computed_after_prediction: boolean;
  };
  feature_motion: "real_support_adaptation" | "unchanged";
  prototype_role: "descriptive_class_centers" | "decision_prototypes";
  calibration: Calibration;
  pool_hash?: string;
  study_id?: string;
  artifact_revision?: string;
  config?: Record<string, unknown>;
  balanced_queries?: boolean;
};
