export type QueryInfo = {
  id: string;
  text: string;
  hops: number;
  start_name?: string;
  relations?: string[];
  sector?: string;
  listed?: boolean;
  graph_id: string;
  split: string;
  seed: number;
  nodes: number;
  edges: number;
  llm_cached: boolean;
  llm_reason: string;
  unavailable_methods: Record<string, string>;
};
export type Metric = {
  split: string;
  budget: number;
  method: string;
  label: string;
  n: number;
  success_rate: number;
  mean_expanded: number;
  mean_hops_success: number | null;
  mean_total_work: number;
  mean_successor_checks: number;
  mean_redundancy: number;
  mean_redundant_paths: number;
  mean_ms: number;
  invalid_answer_rate: number;
  seed_success_sd: number;
};
export type GA = {
  weights: number[];
  history: {
    generation: number;
    fitness: number;
    success_rate: number;
    expanded_fraction: number;
  }[];
  population: number;
  generations: number;
  validation_queries: number;
};
export type Snapshot = {
  warnings: string[];
  remote_available: boolean;
  paired: {
    split: string;
    budget: number;
    method: string;
    success_delta: number;
    graph_bootstrap_low: number;
    graph_bootstrap_high: number;
    graph_clusters: number;
  }[];
  queries: QueryInfo[];
  methods: { id: string; name: string }[];
  summary: Metric[];
  ga: Record<string, GA>;
  learning: {
    seed: number;
    samples: number;
    f1: number;
    brier: number;
    precision: number;
    recall: number;
  }[];
  manifest: {
    runs: number;
    questions: Record<string, number>;
    data_note: string;
  };
  llm_summary: {
    model?: string;
    questions?: number;
    note?: string;
    results?: Metric[];
    total_inference_ms?: number;
    legal_checks?: number;
    legal_check_accuracy?: number;
    explanation_grounded?: number;
    explanations?: number;
  };
  llm_explanations: {
    query_id: string;
    text: string;
    model: string;
    note: string;
    grounded: boolean;
    missing_path_entities: string[];
  }[];
  stress: { source?: string; invalid_answers?: number };
  failures: {
    query: { id: string; text: string };
    method: string;
    status: string;
    diagnosis: string;
    oracle_paths: number;
  }[];
  notes: string[];
};
export type Step = {
  id: number;
  node: string;
  depth: number;
  path: string[];
  edge_ids: string[];
  parent_path: string[];
  status: string;
  g: number;
  h: number | null;
  f: number | null;
  p: number | null;
  reason?: string;
};
export type Node = {
  id: string;
  name: string;
  kind: string;
  sector: string;
  listed: boolean;
};
export type Edge = {
  id: string;
  source: string;
  target: string;
  relation: string;
  confidence: number;
  verified: boolean;
  active: boolean;
  eligible?: boolean;
  rejection_reason?: string;
};
export type Result = {
  query: {
    id: string;
    text: string;
    relations: string[];
    start: string;
    sector: string;
    listed: boolean;
  };
  graph: { nodes: Node[]; edges: Edge[] };
  result: {
    method: string;
    label: string;
    status: string;
    found: boolean;
    valid: boolean;
    answer: string | null;
    path: string[];
    edge_ids: string[];
    expanded: number;
    generated: number;
    hops: number | null;
    total_work: number;
    rollout_steps: number;
    expert_rejections: number;
    redundancy_ratio: number;
    elapsed_ms: number;
    trace: Step[];
    trace_events: number;
    trace_truncated: boolean;
    expert_enabled: boolean;
    classifier_used: boolean;
    heuristic_used: boolean;
    budget: number;
    note: string;
  };
  llm: {
    mode: string;
    note?: string;
    model?: string;
    scored_edges?: number;
    llm_ms?: number;
  };
  path_evidence: Edge[];
  explanation: string;
};
