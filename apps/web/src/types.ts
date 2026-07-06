export type Interval = {
  n: number;
  mean: number;
  standard_deviation: number;
  confidence: number;
  ci_low: number;
  ci_high: number;
};

export type AggregateCheckpoint = {
  step: number;
  retained: Record<string, Interval>;
  adaptation: Record<string, Interval>;
  geometry: Record<string, {
    cosine_centroid_drift: Interval;
    effective_rank: Interval;
    frechet_distance: Interval;
    frechet_projection_dimension: Interval;
    linear_cka: Interval;
    neighborhood_overlap_at_5: Interval;
    class_centroid_movement: Record<string, Interval>;
  }>;
  cross_modal: Record<string, Record<string, Interval>>;
  layerwise: Record<string, Record<string, Record<string, Interval>>>;
  optimization: Record<string, Interval>;
};

export type BenchmarkArtifact = {
  schema_version: string;
  run_id: string;
  config_hash: string;
  evidence_status: string;
  source_manifest: { run_id: string; public_path: string; sha256: string };
  experiment: {
    name: string;
    seeds: number[];
    run_count: number;
    uncertainty: string;
    model: { requested_revision: string; resolved_revision: string; trainable_parameters: number };
    method: Record<string, unknown>;
  };
  checkpoints: AggregateCheckpoint[];
};

export type DetailedCheckpoint = {
  step: number;
  retained: Record<string, number>;
  adaptation: Record<string, number>;
  geometry: Record<string, Record<string, number | Record<string, number>>>;
  classwise: Record<string, {
    class_names: string[];
    confusion_matrix: number[][];
    per_class_accuracy: Record<string, number | null>;
    support: Record<string, number>;
  }>;
  samples: Record<string, Array<{
    id: number;
    label: number;
    baseline: [number, number];
    current: [number, number];
  }>>;
};

export type DetailedArtifact = {
  run_id: string;
  config_hash: string;
  source_manifest: { public_path: string; sha256: string };
  checkpoints: DetailedCheckpoint[];
};

export type EarlyWarningArtifact = {
  evidence_status: string;
  publication_caveat: string;
  evaluation: {
    protocol: {
      split_counts: Record<string, number>;
      maximum_observation_fraction: number;
      selected_ridge_alpha: number;
    };
    test_metrics: Record<string, {
      mae: number;
      rmse: number;
      r_squared: number | null;
      calibration_slope: number | null;
      prediction_interval_empirical_coverage?: number;
    }>;
    test_predictions: Array<{
      scenario_id: string;
      actual_final_forgetting: number;
      predicted_final_forgetting: number;
    }>;
  };
};

export type MethodRecord = {
  id: string;
  label: string;
  category: string;
  fidelity: string;
  strategy: string;
  seeds: number[];
  checkpoints: number[];
  training_budget: {
    probe_steps: number;
    joint_or_adaptation_steps: number;
    total_optimizer_steps: number;
  };
  metrics: {
    baseline_adaptation_accuracy: Interval;
    final_adaptation_accuracy: Interval;
    adaptation_accuracy_change: Interval;
    baseline_retained_accuracy: Interval;
    final_retained_accuracy: Interval;
    retained_accuracy_change: Interval;
    retained_cka_loss: Interval;
    trainable_parameters: Interval;
    adapter_l2_delta: Interval;
  };
};

export type MethodArtifact = {
  schema_version: string;
  run_id: string;
  config_hash: string;
  evidence_status: string;
  publication_caveat: string;
  source_manifest: { run_id: string; public_path: string; sha256: string };
  methods: MethodRecord[];
};

export type InterpolationPoint = {
  alpha: number;
  retained: Record<"top1_accuracy" | "accuracy_change" | "macro_f1", Interval>;
  adaptation: Record<"top1_accuracy" | "accuracy_change" | "macro_f1", Interval>;
  geometry: Record<
    "linear_cka" | "cosine_centroid_drift" | "frechet_distance" | "neighborhood_overlap_at_5",
    Interval
  >;
};

export type InterpolationArtifact = {
  schema_version: string;
  run_id: string;
  config_hash: string;
  evidence_status: string;
  publication_caveat: string;
  source_manifest: { run_id: string; public_path: string; sha256: string };
  experiment: { fidelity: string; seeds: number[]; run_count: number; alphas: number[] };
  curve: InterpolationPoint[];
};

export type DomainScenario = {
  id: string;
  label: string;
  adaptation_domain: string;
  retained_domain: string;
  seeds: number[];
  sample_counts_per_seed: {
    adaptation_eval: number;
    retained_eval: number;
  };
  metrics: {
    adaptation_accuracy_change: Interval;
    retained_accuracy_change: Interval;
    retained_cka_loss: Interval;
    baseline_adaptation_accuracy: Interval;
    final_adaptation_accuracy: Interval;
    baseline_retained_accuracy: Interval;
    final_retained_accuracy: Interval;
  };
};

export type DomainArtifact = {
  schema_version: string;
  run_id: string;
  config_hash: string;
  evidence_status: string;
  publication_caveat: string;
  source_manifest: { run_id: string; public_path: string; sha256: string };
  scenarios: DomainScenario[];
};

export type DatasetExample = {
  src: string;
  label: string;
  row_index: number;
  alt: string;
};

export type DatasetRecord = {
  id: string;
  name: string;
  plain_name: string;
  label_field: string;
  full_size: string;
  native_resolution: string;
  role: string;
  role_explanation: string;
  local_protocol: string;
  source_url: string;
  examples: DatasetExample[];
};

export type DatasetGalleryArtifact = {
  schema_version: string;
  datasets: DatasetRecord[];
};
