export type Method = "baseline" | "maml" | "calibrated";
export type Calibration = "none" | "mean" | "variance" | "prototype" | "full";
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
};
export type LabFourSnapshot = {
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
  training: { epoch: number; loss: number; accuracy?: number }[];
  limitations: string[];
  provenance: { seed: number; checkpoint_sha256: string; run_id?: string };
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
  feature_motion: "real_support_adaptation" | "unchanged";
  prototype_role: "descriptive_class_centers" | "decision_prototypes";
  calibration: string;
};
