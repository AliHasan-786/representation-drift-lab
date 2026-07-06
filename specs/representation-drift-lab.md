# Representation Drift Lab

## Objective

Build a research-grade extension of the original CS 5787 project, "Detecting and Characterizing Representation Drift in Multimodal Vision-Language Models." The completed project will investigate when model adaptation damages prior capabilities, determine which geometric signals reliably predict that damage, compare methods that preserve or recover knowledge, and expose the work through a polished interactive portfolio experience.

The primary audience is recruiters, hiring managers, researchers, engineers, friends, and family. A visitor must be able to understand the core problem within one minute, interact with real experimental results without installing software, and progressively inspect the methodology, evidence, limitations, code, and reproducibility details.

## Context

The legacy project fine-tuned OpenAI CLIP ViT-B/32 on Food-101 using LoRA while treating CIFAR-10 as a retained-capability probe. It measured zero-shot classification, cosine centroid displacement, Frechet distance, confusion matrices, and t-SNE projections.

The audit identified a strong research premise and several shortcomings that the extension must correct:

- The notebook and report visualize a run ending near step 600, while the saved artifact bundle contains 46 checkpoints through step 23,000.
- Dataset sizes, baseline metrics, checkpoint frequencies, and saved fields differ across the notebook, report, and artifact bundle.
- The original evidence uses one backbone, one adaptation method, one source/target pairing, and effectively one seed.
- Several claims use causal language despite correlational evidence.
- The reported early-warning estimate has large error and lacks a held-out validation protocol.
- The training loss treats repeated class captions within a batch as distinct instance pairs, introducing false negatives when multiple images share a class caption.
- The code is a Colab-oriented notebook with runtime package installation, no environment lock, configuration system, test suite, experiment registry, or deployable application.
- The original work was completed by Sahil Mhatre, Ali Hasan, and Corey Chen. The public case study must credit the team for the original project and clearly label the new portfolio extension as Ali Hasan's subsequent independent work.

## Product Principles

- PRIN-001: Scientific traceability takes precedence over visual polish; every displayed number must resolve to a versioned experiment artifact.
- PRIN-002: The experience must support progressive disclosure: plain-language explanation first, technical depth on demand.
- PRIN-003: The public experience must remain functional without a live GPU or third-party inference API.
- PRIN-004: Interactive elements must answer a real research question rather than act as decorative animation.
- PRIN-005: Claims must distinguish observation, association, prediction, and causal evidence.
- PRIN-006: Legacy results remain preserved for historical comparison but must not be silently mixed with newly reproduced results.

## Target User Flows

### Portfolio Visitor

1. Open the project landing page and understand catastrophic forgetting, representation drift, and the project result from a concise visual introduction.
2. Move a training-step control and observe how retained accuracy, target accuracy, embedding geometry, and class confusions change together.
3. Compare adaptation and mitigation methods on a stability-plasticity Pareto frontier.
4. Inspect representative successes and failures.
5. Leave with a clear understanding of Ali's research, engineering, visualization, and product-development contributions.

### Technical Reviewer

1. Inspect exact datasets, splits, models, hyperparameters, seeds, metrics, and hardware for an experiment.
2. Compare results across models, domains, methods, and seeds with uncertainty intervals.
3. Open methodology explanations and primary-source citations.
4. Download compact result tables/configurations and follow reproducibility instructions.

### Interactive Explorer

1. Select a built-in image or upload a supported image.
2. Compare predictions, confidence, nearest examples, and embedding behavior before and after adaptation.
3. Receive a clear fallback experience if live inference is unavailable.

## System Architecture

The repository will use a monorepo-style structure:

- `src/driftlab/`: installable Python package for data, models, training, evaluation, metrics, analysis, and artifact export.
- `configs/`: versioned experiment configurations for smoke, portfolio, and research tiers.
- `tests/`: unit, integration, artifact-contract, and scientific sanity tests.
- `apps/web/`: TypeScript/React portfolio experience, designed as a standalone deployable site and an integration-ready route/module.
- `services/inference/`: optional GPU-backed inference API with no dependency from the core public experience.
- `data/legacy/`: manifests and immutable references to audited original artifacts; large binary assets remain outside Git.
- `artifacts/`: generated local experiment outputs, excluded from Git except for compact approved portfolio artifacts.
- `public/data/`: validated, versioned, web-safe experiment summaries consumed by the frontend.
- `reports/`: generated research summaries, figures, model cards, and reproducibility documentation.

## Research Scope

### Model Families

- REQ-001: The benchmark must include the legacy OpenAI CLIP ViT-B/32 model.
- REQ-002: The benchmark must include at least one reproducible OpenCLIP checkpoint.
- REQ-003: The benchmark must include at least one newer vision-language encoder, with SigLIP 2 preferred subject to compatible licensing and compute.
- REQ-004: A frozen visual encoder such as DINOv2 may be included as a representation-control condition, but its results must not be presented as directly equivalent to image-text zero-shot classification.

### Datasets and Scenarios

- REQ-005: The reproduced legacy scenario must retain Food-101 as the adaptation domain and CIFAR-10 as the historical retained-domain probe.
- REQ-006: The expanded benchmark must include at least three adaptation domains and at least three retained or shifted evaluation domains.
- REQ-007: Dataset selection must cover class, visual-domain, and natural-distribution shifts rather than only unrelated label spaces.
- REQ-008: Every dataset split must be deterministic, fingerprinted, documented, and isolated so evaluation samples never influence optimization or model selection.
- REQ-009: Dataset licenses, redistribution constraints, provenance, preprocessing, and known limitations must be documented.
- REQ-010: The pipeline must provide smoke-test subsets that execute without downloading the full research corpus.

### Adaptation and Mitigation Methods

- REQ-011: Baselines must include zero-shot evaluation, linear probing, full fine-tuning, and standard LoRA.
- REQ-012: Robust adaptation comparisons must include LP-FT and WiSE-FT.
- REQ-013: Continual-learning comparisons must include at least one distillation/reference-data method inspired by ZSCL or Learning without Forgetting.
- REQ-014: Parameter-efficient comparisons must include at least one selective, adaptive-rank, or probabilistic LoRA variant.
- REQ-015: Geometry-preserving comparisons must include a null-space or gradient-projection strategy.
- REQ-016: Recovery comparisons must include a post-hoc merge or interpolation method such as MergeTune when implementation compatibility permits.
- REQ-017: Every paper-derived method must be labeled as an exact reproduction, adapted implementation, or inspired baseline; the project must not imply reproduction fidelity without verification.
- REQ-018: The loss implementation must correctly handle duplicate class captions and must include a test demonstrating that same-class examples are not incorrectly treated as false negatives under class-supervised training.

### Metrics and Analysis

- REQ-019: Task metrics must include top-1 accuracy, macro F1, negative log-likelihood, and expected calibration error where applicable.
- REQ-020: Continual-learning metrics must include average accuracy, final accuracy, forgetting, backward transfer, forward transfer where defined, and retained zero-shot performance.
- REQ-021: Stability-plasticity reporting must show both raw metrics and normalized changes from each model's baseline.
- REQ-022: Geometric diagnostics must include cosine centroid displacement, a numerically stable distribution-distance measure, linear CKA, covariance/effective-rank analysis, class-centroid movement, and local-neighborhood overlap.
- REQ-023: Frechet-style distance must use documented covariance regularization or dimensionality reduction and must be tested for finite, real, stable output under rank-deficient sample covariance.
- REQ-024: Layerwise analysis must identify where drift emerges across the visual encoder instead of measuring only final embeddings.
- REQ-025: Cross-modal analysis must separately track image-space drift, text-space drift, and image-text alignment when both modalities can change.
- REQ-026: Dimensionality-reduction plots must use a fixed, reproducible projection or alignment strategy across checkpoints; independent t-SNE plots may not be interpreted as physical trajectories.
- REQ-027: All primary comparisons must run with at least three random seeds in the portfolio tier and report 95% uncertainty intervals.
- REQ-028: Drift-performance relationships must report Pearson and Spearman association with confidence intervals and must not use causal language without an intervention-based design.
- REQ-029: Early-warning models must use temporally valid train/validation/test separation across runs or task scenarios, compare against naive baselines, and report calibration and prediction error.
- REQ-030: The analysis must include failure cases where geometric drift does not correspond to forgetting or where performance changes without large measured drift.

## Reproducibility and Provenance

- REQ-031: The Python project must use a declared supported Python version, `pyproject.toml`, and a committed dependency lock.
- REQ-032: Every run must record a unique run ID, configuration hash, Git commit, seed, dataset fingerprints, model revision, environment, hardware, start/end timestamps, and artifact schema version.
- REQ-033: Training must be configuration-driven and runnable from a non-notebook CLI.
- REQ-034: Runs must support safe resume from checkpoints without duplicating steps or overwriting unrelated artifacts.
- REQ-035: Metrics and web artifacts must be regenerated from saved model outputs by deterministic commands.
- REQ-036: Large checkpoints and embeddings must not be committed to Git; manifests must document their storage location and checksums.
- REQ-037: A legacy reconciliation report must map each original claim and figure to its source artifact or explicitly mark it unverified.
- REQ-038: The repository must provide CPU smoke tests and clearly separated GPU research commands.
- REQ-039: A clean environment must be able to reproduce the smoke pipeline from documented commands.

## Interactive Portfolio Experience

- REQ-040: The landing section must explain the problem, intervention, primary finding, and project scope in plain language without requiring scrolling through methodology first.
- REQ-041: The page must include a training timeline control that synchronizes retained accuracy, target accuracy, drift metrics, and embedding/class views.
- REQ-042: The page must include a stability-plasticity Pareto explorer for comparing models and methods.
- REQ-043: The page must include a layerwise drift view using an interactive heatmap or architecture diagram.
- REQ-044: The page must include a class microscope showing per-class accuracy, confusion, centroid movement, and representative samples.
- REQ-045: The page must include a method-comparison matrix covering performance, forgetting, drift, parameter count, training cost, and inference cost.
- REQ-046: The page must include an early-stopping simulator that lets users select a drift threshold and observe the resulting retained/adapted performance trade-off using recorded checkpoints.
- REQ-047: The page must include representative before/after predictions and explicitly curated failure cases.
- REQ-048: Every chart must expose definitions, units, sample sizes, uncertainty, and the exact run/configuration behind the displayed values.
- REQ-049: The experience must provide separate plain-language and technical-depth paths without duplicating contradictory content.
- REQ-050: The original team, Ali's original role where documented, and Ali's independent extension must be attributed clearly.
- REQ-051: The site must include methodology, limitations, bibliography, reproducibility, and project-history sections.
- REQ-052: The site must include concise resume-ready outcomes without inflating unverified research claims.

### Live Inference

- REQ-053: Core charts and interactions must use precomputed, versioned artifacts and work when the inference service is offline.
- REQ-054: The optional inference service must support a constrained built-in gallery and user-upload flow for selected compatible models.
- REQ-055: Uploads must enforce file type and size limits, strip unnecessary metadata, avoid permanent storage by default, and display the data-handling policy.
- REQ-056: The frontend must show service availability, progress, timeout, rate-limit, and failure states without blocking the rest of the case study.
- REQ-057: Live outputs must be labeled separately from benchmark results and may not alter published experiment artifacts.

## Design, Accessibility, and Performance

- REQ-058: The experience must be responsive across current mobile, tablet, laptop, and wide-desktop layouts.
- REQ-059: All functionality must be keyboard accessible and meet WCAG 2.2 AA color-contrast and focus requirements.
- REQ-060: Charts must provide text summaries, accessible labels, non-color encodings, and usable tooltips.
- REQ-061: Motion must respect `prefers-reduced-motion`; no essential information may depend solely on animation.
- REQ-062: Initial portfolio content and primary charts must load without downloading model checkpoints or full embeddings.
- REQ-063: Large visualization data must be compressed, lazy-loaded, sampled, or tiled so interaction remains responsive.
- REQ-064: The production build must define and pass explicit performance budgets for JavaScript, data payloads, layout stability, and interaction latency.
- REQ-065: The visual system must feel like a finished portfolio case study rather than a notebook or generic analytics dashboard.

## Testing and Validation

- REQ-066: Metric implementations must have unit tests against hand-computed or trusted small examples.
- REQ-067: Dataset split, seed, checkpoint resume, artifact schema, and configuration hashing behavior must have automated tests.
- REQ-068: At least one end-to-end smoke experiment must train, evaluate, export artifacts, and render in the web application in CI-compatible time.
- REQ-069: Web tests must cover the primary visitor flow, checkpoint scrubber, comparison controls, error states, and responsive layouts.
- REQ-070: Accessibility must be checked automatically and manually for keyboard navigation and screen-reader structure.
- REQ-071: Published numerical claims must be validated against their source artifacts during the production build.
- REQ-072: The final report must include limitations, negative results, compute usage, and known threats to validity.

## Research Sources

The implementation and narrative must prioritize primary sources, including:

- CLIP: <https://arxiv.org/abs/2103.00020>
- LoRA: <https://arxiv.org/abs/2106.09685>
- Feature distortion and LP-FT: <https://arxiv.org/abs/2202.10054>
- WiSE-FT: <https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html>
- ZSCL: <https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Preventing_Zero-Shot_Transfer_Degradation_in_Continual_Learning_of_Vision-Language_Models_ICCV_2023_paper.html>
- CKA: <https://proceedings.mlr.press/v97/kornblith19a>
- CLAP4CLIP: <https://openreview.net/forum?id=rF1YRtZfoJ>
- NuSA-CL: <https://openreview.net/forum?id=tucuU4sQ3s>
- MergeTune: <https://openreview.net/forum?id=MAApSY32Z6>
- SigLIP 2: <https://arxiv.org/abs/2502.14786>

## Sprint Plan

### Sprint 0 - Audit and Research

- Inventory the course archive and legacy artifacts.
- Audit code, data, metrics, report claims, figures, and checkpoint contents.
- Identify relevant course methodology and current research directions.

### Sprint 1 - Foundation and Reconciliation

- Scaffold the repository, environment, schemas, CLI, configuration system, and tests.
- Import legacy artifacts through immutable manifests.
- Produce the legacy reconciliation report.
- Implement a small deterministic smoke pipeline.

### Sprint 2 - Reproduction

- Reproduce the original CLIP/Food-101/CIFAR-10 experiment under a traceable configuration.
- Correct the duplicate-caption loss issue.
- Compare reproduced outputs with each legacy result family.
- Establish the validated baseline used by the public narrative.

### Sprint 3 - Diagnostics

- Implement robust task, continual-learning, calibration, and geometry metrics.
- Add layerwise, classwise, local-neighborhood, and cross-modal analysis.
- Add uncertainty estimation and statistically valid association analysis.
- Build and validate the early-warning protocol.

### Sprint 4 - Mitigation Benchmark

- Run baseline adaptation methods.
- Implement and compare robust, continual, geometry-preserving, and recovery methods.
- Run multiple seeds and selected model/domain combinations.
- Select representative results and negative cases for publication.

### Sprint 5 - Portfolio Application

- Build the narrative landing experience.
- Implement checkpoint, embedding, class, layerwise, Pareto, comparison, and early-stop interactions.
- Integrate validated web artifacts and source-level provenance.
- Add optional live inference with fallback behavior.

### Sprint 6 - Publication Quality

- Complete technical writing, attribution, bibliography, model/data documentation, and resume copy.
- Validate claims, accessibility, responsive design, performance, and production builds.
- Produce integration instructions and a deployable standalone route/site.

### Sprint 7 - Website Integration

- Integrate the project into the personal website when its repository is available, or deploy the standalone experience and connect it through a portfolio project route/link.
- Verify analytics, metadata, social preview, canonical URL, and production behavior.

## Constraints

- CON-001: The build may scale substantially, but published claims must be limited to experiments actually completed and validated.
- CON-002: The public application must not depend on Cornell-restricted lecture content or redistribute course materials.
- CON-003: Original teammates must be credited and the extension must not imply sole authorship of the 2025 group project.
- CON-004: Dataset and model licenses must be respected; restricted raw data must not be bundled into the public repository or deployment.
- CON-005: Secrets, API keys, private paths, personal identifiers beyond intentional attribution, and local machine metadata must not enter committed artifacts.
- CON-006: The core site must have a static/precomputed mode; live GPU inference is an enhancement, not a single point of failure.
- CON-007: Research-tier experiments may require external GPU compute, but smoke and artifact-validation workflows must remain locally testable.
- CON-008: Until the personal website repository is supplied, the web experience will be built as a standalone integration-ready application with a documented embedding/linking contract.

## Edge Cases

- EDGE-001: If a model or dataset cannot be redistributed, publish aggregate artifacts and instructions rather than restricted files.
- EDGE-002: If a research method is incompatible with a selected backbone, mark the matrix cell unsupported and explain why rather than silently substituting another method.
- EDGE-003: If fewer than three valid seeds complete, exclude confidence claims and label results preliminary.
- EDGE-004: If drift metrics disagree, show the disagreement and avoid collapsing them into an unjustified composite score.
- EDGE-005: If live inference is unavailable or times out, retain the uploaded image only in browser memory where possible, show a clear fallback, and keep all precomputed exploration usable.
- EDGE-006: If an uploaded file is invalid, oversized, animated, corrupted, or unsupported, reject it before inference with actionable feedback.
- EDGE-007: If an experiment artifact fails schema or provenance validation, the production build must reject it rather than display partial data.
- EDGE-008: If legacy values cannot be traced to source code, preserve them only in a clearly labeled historical section.
- EDGE-009: If a visualization would require misleading alignment across independently fitted projections, replace it with a shared projection or explicitly non-trajectory view.
- EDGE-010: Empty comparison selections and narrow-screen layouts must retain explanations and recovery controls.

## Out of Scope

- Training a foundation vision-language model from scratch.
- Reproducing every lecture topic or homework as a separate portfolio feature.
- Claiming production safety or universal catastrophic-forgetting prevention.
- Publishing restricted course slides, assignment instructions, teammate data, or unlicensed datasets.
- Allowing anonymous arbitrary-duration training jobs from the public website.

## Definition of Done

- DONE-001: A clean checkout can install the locked environment and run the documented CPU smoke pipeline successfully.
- DONE-002: The legacy reconciliation report identifies the source and validation status of every original headline result and figure.
- DONE-003: The reproduced legacy scenario is traceable to exact code, config, seed, model revision, dataset fingerprints, and artifacts.
- DONE-004: The benchmark includes at least three model families or control families, three adaptation domains, three retained/shifted domains, and the required method categories.
- DONE-005: Portfolio-tier primary comparisons contain at least three seeds and uncertainty intervals.
- DONE-006: All required metrics pass automated correctness and numerical-stability tests.
- DONE-007: The early-warning evaluation uses held-out temporal/scenario validation and reports performance against naive baselines.
- DONE-008: The public experience implements the synchronized checkpoint explorer, Pareto view, layerwise view, class microscope, method matrix, early-stop simulator, and failure-case gallery.
- DONE-009: Every published chart and numerical claim resolves to a validated versioned artifact.
- DONE-010: The core public experience works with JavaScript enabled and the live inference service unavailable.
- DONE-011: The application passes production build, responsive-browser, accessibility, and agreed performance-budget checks.
- DONE-012: Original-team attribution, extension ownership, dataset/model licenses, limitations, and primary-source citations are visible and accurate.
- DONE-013: The repository contains clear setup, experiment, artifact-generation, testing, deployment, and website-integration documentation.
- DONE-014: A recruiter can understand the problem, Ali's contribution, the primary evidence, and the engineering scope within a three-minute guided path.
- DONE-015: The final project is deployed as a standalone portfolio experience or integrated route and is connected to Ali's personal website.
