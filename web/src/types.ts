export type Entity = {
  eid: string;
  name: string;
  kind: "Person" | "Company";
  category: string;
  capital: number;
  opacity: number;
  disclosure: number;
};
export type Holding = {
  holder: string;
  target: string;
  ratio: number;
  since: string;
  layered: boolean;
};
export type CaseData = {
  schema_version: 1;
  entities: Entity[];
  holdings: Holding[];
  guarantees: { guarantor: string; beneficiary: string; amount: number }[];
  trades: { seller: string; buyer: string; amount: number }[];
  kinships: { a: string; b: string; relation: string; evidence: boolean }[];
  source: string;
  label: string;
};
export type Completeness = {
  holdings: boolean;
  trades: boolean;
  guarantees: boolean;
  registry: boolean;
};
export type Parameters = {
  method: string;
  threshold: number;
  model_source: string;
};
export type CaseBundle = {
  id: string;
  case: CaseData;
  completeness: Completeness;
};
export type Bootstrap = {
  cases: { id: string; name: string; detail: string }[];
  models: { id: string; name: string; detail: string }[];
  methods: { id: string; name: string }[];
  default_case: string;
  default_model: string;
  limits: { entities: number; relations: number; bytes: number };
};
export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  kind: "holding" | "guarantee" | "trade" | "kinship" | "control";
  label: string;
  fact_id: string;
  index: number;
};
export type FactNode = {
  id: string;
  fact: (string | number)[];
  label: string;
  rule: string | null;
  rule_description: string;
  premises: string[];
  source: string;
};
export type Decision = {
  deferred_structural_alert?: boolean;
  minimum_support?: number;
  company_id: string;
  company_name: string;
  score: number;
  risk_class: number;
  review_required: boolean;
  review_priority: "低" | "中" | "高";
  lock_changed_action: boolean;
  constraint_violation: boolean;
  coverage: number;
  alerts: string[];
  controllers: string[];
  claim_ids: string[];
  components: Record<string, number>;
  weights: Record<string, number>;
  observation: { values: Record<string, number | null> };
  hybrid_comparison: {
    unlocked: {
      score: number;
      review_required: boolean;
      review_priority: string;
    };
    locked: {
      score: number;
      review_required: boolean;
      review_priority: string;
    };
  };
};
export type ControlQuery = { source: string; target: string };
export type Analysis = {
  id: string;
  case: CaseData;
  completeness: Completeness;
  parameters: Parameters;
  entities: Entity[];
  edges: GraphEdge[];
  derived_edges: GraphEdge[];
  decisions: Record<string, Decision>;
  provenance: Record<string, FactNode>;
  steps: {
    cycle: number;
    rule: string;
    fact_ids: string[];
    description: string;
  }[];
  stats: {
    cycles: number;
    facts: number;
    derived_facts: number;
    candidate_checks: number;
    elapsed_ms: number;
    preparation_ms: number;
    entities: number;
    companies: number;
    persons: number;
    relations: number;
  };
  model: { weights: Record<string, number>; bayes: { prior: number } };
  default_company: string;
  wall_ms: number;
  cached: boolean;
  created_at: string;
};
export type RuleItem = {
  name: string;
  description: string;
  group: string;
  priority: number;
  antecedents: string[];
  consequent: string;
  guard_variables: string[];
  guard_conditions: string[];
  guard_complete: boolean;
  has_guard: boolean;
  note: string;
};
export type QueryResult = {
  goal: string[];
  proved: boolean;
  equivalent: boolean;
  lines: string[];
  stats: Record<string, number>;
  message: string;
};
export type ExperimentRow = Record<string, string | number | boolean | null>;
export type ExperimentsData = {
  available: boolean;
  metadata: {
    created_utc?: string;
    config?: {
      seeds: number[];
      test_size: number;
      train_size: number;
      validation_size: number;
    };
  };
  summary: ExperimentRow[];
  paired: ExperimentRow[];
  benchmark: ExperimentRow[];
  indicator_summary: ExperimentRow[];
  sensitivity: ExperimentRow[];
  conflict_summary: ExperimentRow[];
  goal_slicing: ExperimentRow[];
  resolution: ExperimentRow[];
  note: string;
};

export type LabId = "lab01" | "lab02";
export type CourseModule = {
  id: string;
  title: string;
  requirement: string;
  description: string;
  kind: string;
};
export type CourseLab = {
  id: LabId;
  number: string;
  name: string;
  summary: string;
  code: string;
  entry: string;
  modules: CourseModule[];
};
export type CourseCatalog = {
  title: string;
  scene: string;
  labs: CourseLab[];
  note: string;
};
export type LabOneResult = {
  experiment: string;
  version: string;
  entities: { eid: string; name: string; kind: string }[];
  triples: {
    source: string;
    target: string;
    relation: string;
    name: string;
    value: string | number;
  }[];
  schema: {
    id: string;
    name: string;
    head_types: string[];
    tail_types: string[];
    attribute: string;
    inferred: boolean;
  }[];
  facts: (string | number)[][];
  rules: {
    name: string;
    description: string;
    conditions: (string | number)[][];
    conclusions: (string | number)[][];
    priority: number;
    has_guard: boolean;
    guard_conditions: string[];
    guard_complete: boolean;
  }[];
  frames_before: FrameResult[];
  frames_after?: FrameResult[];
  stats: Record<string, number>;
  elapsed_ms: number;
  note: string;
  goal?: (string | number)[];
  proved?: boolean;
  proof?: string[];
  steps?: {
    cycle: number;
    rule: string;
    fact: (string | number)[];
    premises: (string | number)[][];
  }[];
  controls?: (string | number)[][];
  alerts?: (string | number)[][];
  derived_facts?: (string | number)[][];
  enhanced_relations?: { source: string; target: string; rule: string }[];
  owners?: {
    owner: string;
    name: string;
    ratio: number;
    beneficial: boolean;
    paths: [string[], number][];
  }[];
  path?: [string, string, string, string][] | null;
  holding_cycles?: string[][];
  guarantee_cycles?: string[][];
  target?: string;
  source?: string;
  benchmark?: {
    entities: number;
    holdings: number;
    seed: number;
    repeats: number;
    results: {
      depth: number;
      method: string;
      median_ms: number;
      min_ms: number;
      max_ms: number;
    }[];
  } | null;
};
export type FrameResult = {
  id: string;
  name: string;
  parent: string | null;
  slots: { name: string; value: unknown; unit: unknown; inherited: boolean }[];
};
export type LabTwoResult = {
  model_source?: string;
  decision_threshold?: number;
  timing_scope?: string;
  task: string;
  facts?: { fact: (string | number)[]; formula: string; source: string }[];
  fact_counts?: Record<string, number>;
  derived_facts?: {
    fact: (string | number)[];
    rule: string;
    premises: (string | number)[][];
  }[];
  source?: string;
  target?: string;
  goal?: string | (string | number)[];
  proved?: boolean;
  status?: string;
  equivalent?: boolean;
  stats?: Record<string, number>;
  forward_stats?: Record<string, number>;
  backward_stats?: Record<string, number>;
  proof?: string[];
  forward_proof?: string[];
  backward_proof?: string[];
  input_facts?: number;
  steps?: { cycle: number; rule: string; facts: (string | number)[][] }[];
  observation?: {
    values: Record<string, number | null>;
    coverage?: number;
    source: string;
  };
  bayes?: {
    score: number;
    parameters: Record<string, unknown>;
    elapsed_ms: number;
  };
  fuzzy?: {
    score: number;
    parameters: Record<string, unknown>;
    explanation: Record<string, unknown>;
    elapsed_ms: number;
  };
  decisions?: (Decision & {
    method: string;
    proofs: { fact: (string | number)[]; lines: string[] }[];
  })[];
  elapsed_ms: number;
  note: string;
};
