# Legacy Result Reconciliation

## Purpose

This report separates the original 2025 group-project record from the independently rebuilt portfolio extension. It records which legacy claims can be traced to saved evidence and which require reproduction before publication as validated results.

Original project team: Sahil Mhatre, Ali Hasan, and Corey Chen. The historical report, presentation, notebook, and artifacts are shared team work. The new repository and experiments are Ali Hasan's subsequent independent extension.

## Evidence Families

### Family A - Notebook/report run

- Source: `DeepLearningFinalProject.ipynb`, its Markdown export, and five exported notebook figures.
- Visible configuration: CLIP ViT-B/32, Food-101 adaptation, CIFAR-10 retention probe, LoRA rank 8, alpha 16, dropout 0.1, batch size 32, five epochs, checkpoint frequency 200.
- Visible data: 5,000 Food-101 training samples, 1,000 Food-101 evaluation samples, and 1,000 CIFAR-10 evaluation samples.
- Executed output: 785 optimization steps and recorded figures through step 600.
- Baseline output: CIFAR-10 accuracy 0.9000 and Food-101 accuracy 0.7520.
- Status: traceable to notebook output, but not independently reproduced.

### Family B - Saved 23,000-step bundle

- Source: `drift_analysis_final_submission_v2` artifact bundle.
- Verified contents: 46 paired checkpoint embedding files from steps 500 through 23,000, two baselines, a full model state, a metric table, two figures, and a short summary.
- Verified embedding shapes: CIFAR-10 has 2,000 samples by 512 dimensions; Food-101 has 1,000 samples by 512 dimensions. Inspected embeddings are finite, unit-normalized, and retain consistent labels.
- Verified model state: 446 tensors, including 48 LoRA tensors totaling 294,912 values, stored with the full CLIP state.
- Metric-table implied baselines: CIFAR-10 accuracy 0.9045 and Food-101 accuracy 0.7520.
- Final reported values at step 23,000: CIFAR-10 accuracy 0.8130, Food-101 accuracy 0.7730, cosine drift 0.3055, and Frechet distance 0.6428.
- Status: values are readable and checksum-registered, but the producing training code/configuration is absent.

### Family C - Traceable local reproduction validation

- Source: `configs/reproduction-local.yaml`, run `clip-food101-cifar10-local-a3410b237d0c5b79-s42`, and `public/data/reproduction-local.json`.
- Model: `openai/clip-vit-base-patch32` resolved to commit `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
- Method: LoRA on vision-attention `q_proj` and `v_proj`, rank 8, alpha 16, dropout 0.1, with exactly 294,912 trainable parameters and a class-aware multi-positive contrastive loss.
- Data: deterministic dataset-server selections pinned to Food-101 commit `83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9` and CIFAR-10 commit `0b2714987fa478483af9968de7c934580d0bb9a2`. The validation tier contains six Food-101 classes with 24 training and 24 disjoint evaluation images plus 30 CIFAR-10 evaluation images.
- Training: one MPS run, seed 42, 20 steps, batch size 4, learning rate 5e-4, checkpoints at steps 0, 5, 10, and 20.
- Observed adaptation accuracy: 0.7917 at baseline and 0.8750 at step 20.
- Observed retained accuracy: 0.7333 at baseline and 0.7333 at step 20, with a temporary increase to 0.7667 at step 10.
- Representation diagnostics: retained linear CKA declined from 1.0 to 0.9882 while the LoRA adapter moved 1.9803 in L2 norm from initialization. The artifact also includes fixed-basis geometry, every vision-transformer block, cross-modal alignment, and classwise errors.
- Provenance: public artifact SHA-256 `1387498f545eea93fe687846bd4aec0e8f2261c4409d89b6132ebf5bcebf4b8d`; deployable manifest SHA-256 `156d59a2cc2175841ffa236de2ae0e164ac9f92d7f423dad651d3ffced4975b4`.
- Status: reproducible pipeline validation, explicitly preliminary. Its tiny stratified subset, elevated learning rate, short duration, and single seed do not establish the legacy headline result.

### Family D - Local independent-seed protocol validation

- Source: `configs/reproduction-local-multiseed.yaml`, three independent runs at seeds 41, 42, and 43, and `public/data/benchmark-local.json`.
- Final mean retained accuracy: 0.7000 with a two-sided 95% Student-t interval of [0.6172, 0.7828].
- Final mean adaptation accuracy: 0.8889 with a two-sided 95% Student-t interval of [0.8291, 0.9486].
- Final retained linear CKA: 0.9874 with a two-sided 95% Student-t interval of [0.9853, 0.9895].
- Provenance: aggregate public artifact SHA-256 `3699f2d6b41351d50d51a58958614a0d892d73676b18f419bf8350d0022c2331`; deployable manifest SHA-256 `ec202de88a21289bbe5187f8a28e38df3f8adfb8fc67525cc14d2a78f01e1f75`.
- Status: independent-seed and uncertainty machinery validated, but still explicitly preliminary because the sample and scenario are intentionally bounded. It is not a substitute for the portfolio-tier benchmark.

### Family E - Local multi-method protocol validation

- Source: nine three-seed method suites composed by `configs/method-comparison-local.yaml` into `public/data/method-comparison-local.json`.
- Measured interventions: frozen linear probing, random-head full fine-tuning, LP-FT, zero-shot-initialized full fine-tuning, adapted WiSE-FT, standard LoRA, ZSCL-inspired retention distillation, first-order retention-gradient null-space projection, and selective value-projection LoRA.
- Mean final adapted accuracy ranged from 0.8889 to 1.0000; mean retained accuracy ranged from 0.6889 to 0.7333. The linear probe's perfect adapted score is explicitly treated as tiny-subset separability, not a realistic Food-101 estimate.
- Negative result: across the nine intervention means, retained CKA loss and forgetting had Pearson association 0.5447, Spearman association 0.2594, and a 95% bootstrap Pearson interval [-0.7896, 0.9695]. The local experiment does not support treating CKA loss as a reliable method-ranking proxy.
- Provenance: aggregate public artifact SHA-256 `151bab191b05b2a8702c5de0d6ebe7f13d4e65933c214b3f0c1313549535360a`; its embedded deployable manifest checksum is `9ac6b060f935a0417456a872ce032be254d8df104dae8319bac0b8027fcd2d17`.
- Status: measured mitigation protocol, explicitly preliminary. Paper-derived names are qualified as inspired baselines rather than exact reproductions.

### Family F - Historical source-code scaffold

- Source: external folder `DL Final`, registered in `data/legacy/code-archive-manifest.json` and audited in `reports/legacy-code-audit.md`.
- Verified contents: a 46,659-byte ZIP and extracted 2025 code tree covering CLIP/LoRA training, embedding extraction, drift metrics, visualization, tests, notebooks, and several explicit placeholders. The extracted README is a later expansion of the README inside the ZIP.
- Intended configuration: CLIP ViT-B/32, rank-8/alpha-16 LoRA on vision `q_proj` and `v_proj`, Food-101 adaptation, primarily COCO embedding evaluation, and cosine/centroid/Frechet/alignment diagnostics.
- Missing evidence: no datasets, checkpoints, embeddings, results, logs, executed notebook cells, or run-specific provenance.
- Audit result: 32 Python files parse and three shell scripts pass syntax checking, but the advertised pipeline has incompatible CLI calls, missing test imports, incomplete checkpoint reconstruction, placeholder phases, false-negative-prone supervision, and an early-prediction leakage path.
- Status: historical design and code-lineage evidence only. It is not the producing code for Families A or B and contributes no new measured result.

## Headline Claim Map

| Claim | Source evidence | Status | Publication treatment |
| --- | --- | --- | --- |
| Food-101 improves while CIFAR-10 declines | Families A and B | Partially corroborated | Present as a legacy observation until reproduced across seeds |
| Food-101 gain is 2.10 percentage points | Family B | Traceable to artifact | Label historical; do not merge with Family A |
| CIFAR-10 loss is 9.15 percentage points | Family B | Traceable to artifact | Label historical; do not merge with Family A |
| Drift rises during adaptation | Family B table and both plot families | Traceable association | Describe as an observed trend, not a mechanism |
| Frechet distance proves or causes forgetting | Qualitative overlay only | Unsupported | Remove causal/proof language |
| Step-8,000 drift predicts final drift | Predicted 0.6647 versus actual 0.3055 | Failed predictor | Publish as a motivating negative result |
| Drift follows a coherent semantic direction | Family A joint t-SNE | Suggestive only | Re-test with fixed projections and paired metrics |
| Cat/dog and bird/frog are especially damaged | Family A confusion plot and prose | Not tabulated | Recompute before publication |
| The corrected runner can adapt CLIP using only LoRA parameters | Family C manifest and checkpoint diagnostics | Reproduced locally | Use as engineering validation, not a benchmark claim |
| Three-seed aggregation reports independent-run uncertainty | Family D manifests and aggregate artifact | Reproduced locally | Use as protocol validation; retain the preliminary label |
| Lower final CKA loss ranks methods by retention | Family E intervention means | Not supported | Publish the disagreement as a negative result |

## Figure Map

| Figure | Evidence family | Validation status |
| --- | --- | --- |
| Stability-plasticity plot in final report | Family A, steps 0-600 | Source identified; reproduction pending |
| Geometry plot in final report | Family A, steps 0-600 | Source identified; reproduction pending |
| Drift-versus-forgetting plot | Family A, steps 0-600 | Source identified; causal interpretation rejected |
| Confusion matrix | Family A | Source identified; numerical table absent |
| t-SNE trajectories | Family A | Source identified; projection interpretation limited |
| Artifact-bundle tradeoff plot | Family B, steps 500-23,000 | Checksum registered; producing code absent |
| Artifact-bundle geometry plot | Family B, steps 500-23,000 | Checksum registered; producing code absent |

## Inconsistencies Requiring Reproduction

1. The report describes 74,750 Food-101 training images and 2,000 CIFAR-10 evaluation images; Family A code selects 5,000 and 1,000.
2. Family A can produce 785 steps; Family B extends to 23,000.
3. Family A checkpoints every 200 steps; Family B artifacts every 500.
4. Family B contains text embeddings; Family A extraction saves only image embeddings and labels.
5. Family A baseline CIFAR-10 accuracy is 0.9000; Family B arithmetic implies 0.9045.
6. The report has a broken section reference, a `400 epochs` typo, and ACM placeholders.
7. Family F uses COCO as its primary retention/evaluation path and has no CIFAR-10 loader, so it does not explain the CIFAR-10 report/notebook run.

## Reproduction Gate

No legacy number becomes a primary portfolio claim until a new run records exact configuration, code revision, model revision, dataset fingerprints, seed, environment, checkpoints, and metrics. Legacy values may appear only in a labeled project-history section linked to this report.

Families C through E satisfy the traceability, deterministic regeneration, independent-seed, uncertainty, and initial intervention mechanisms. The gate for primary scientific claims remains closed until the larger protocol covers representative model, method, adaptation-domain, and retention-domain combinations.
