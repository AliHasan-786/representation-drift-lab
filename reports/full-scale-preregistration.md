# Preregistration: Full-Scale GPU Benchmark (Sprint A of the research roadmap)

Registered before execution, per the roadmap's non-negotiable rules. This
document locks the design; results are published whether or not they
preserve the current narrative.

## Question

Does the Food-101 → CIFAR-10 adaptation/retention trade-off measured at the
expanded local tier persist when the evaluation is large enough to escape
the documented ceiling effects (perfect tiny-subset probe scores, wide
Student-t intervals)?

## Design (locked)

- Config: `configs/full-scale-gpu.yaml` via
  `configs/full-scale-gpu-multiseed.yaml`.
- Adaptation: Food-101, 32 classes × 16 train / 16 held-out eval images
  per class (512 train, 512 eval), stratified from the pinned revision.
- Retention: CIFAR-10 test split rows 1000–1499 (500 images), disjoint
  from every training, reference, and prior evaluation range.
- Method: standard LoRA (rank 8, q/v projections), 200 steps, batch 16,
  checkpoints 0/50/100/200 — the same intervention family as the local
  tier so results are comparable, not a new method.
- Seeds: 201–205 (five independent seeds; intervals remain 95% Student-t).
- Compute: free tier only — Kaggle GPU (~30 h/week) or Colab, via
  `kaggle/full-scale-benchmark.ipynb`. No paid infrastructure.

## Primary metrics (locked before scoring)

1. Final adaptation accuracy (Food-101 held-out) with CI.
2. Final retained accuracy change (CIFAR-10 probe) with CI.
3. Retained 1 − CKA at the final checkpoint.

Secondary/diagnostic: layerwise drift profile, cross-modal alignment,
calibration (ECE), effective rank. Secondary metrics cannot promote a
conclusion on their own.

## Promotion gate

Published to the portfolio benchmark table only if:

- all five seeds complete with validating manifests;
- the frozen and zero-shot controls reproduce within tolerance;
- disjointness checks pass for every split (no score-bearing image
  influenced any update).

If the expanded ranking disagrees with the local tier, the disagreement is
the finding and both tables stay published.

## Status

- [x] Design locked and committed before execution
- [ ] Seeds 201–205 executed (Kaggle notebook)
- [ ] Manifests validated, controls reproduced
- [ ] Results published to `public/data/benchmark-full-scale-gpu.json`
