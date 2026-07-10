# Representation Drift Lab

Representation Drift Lab extends the original Cornell Tech CS 5787 project on catastrophic forgetting in CLIP into a reproducible benchmark, mitigation laboratory, and interactive portfolio experience.

The original 2025 project was completed by Sahil Mhatre, Ali Hasan, and Corey Chen. This repository is Ali Hasan's independent post-course extension. Historical results are preserved as historical evidence; new claims must pass the provenance and validation gates defined in the project specification.

## Current status

- Course archive and legacy artifacts audited.
- Legacy evidence registered by checksum and reconciled in `reports/legacy-reconciliation.md`.
- A separately supplied historical source scaffold is checksum-registered and audited in `reports/legacy-code-audit.md`; it is treated as code lineage, not new experimental evidence.
- Three domain pairs pass the current publication gate. A fourth exploratory Pets-to-EuroSAT artifact is preserved in `reports/excluded-runs.md` but excluded because its retained baseline was 0% in every seed.
- Configuration, provenance, metric, artifact, resume, and CPU smoke foundations implemented.
- A traceable CLIP/LoRA reproduction tier has completed against pinned Food-101 and CIFAR-10 revisions with final, layerwise, classwise, and cross-modal diagnostics.
- Nine local adaptation/mitigation methods have completed across three independent seeds with 95% Student-t intervals, spanning frozen probing, full fine-tuning, LP-FT, adapted WiSE-FT, LoRA, distillation, selective adaptation, and gradient projection.
- The standalone portfolio application is implemented, tested responsively, artifact-gated, and production-built. It now includes a zero-assumption explainer, real examples from all six experiment datasets, an interactive output-reading tutorial, the exact original course report, and a grounded project Q&A guide with an offline fallback. Its current evidence remains explicitly preliminary while broader model/domain experiments continue.

The authoritative requirements are in `specs/representation-drift-lab.md`.

## Environment

Supported Python versions: 3.11 through 3.14.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

The foundation lock contains the deterministic smoke dependencies. Real-model dependencies are separately pinned in `requirements.research.lock`.

Install the separately locked real-model stack when running CLIP experiments:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.research.lock
```

## Run the CPU smoke pipeline

```bash
make smoke
make validate
```

The smoke pipeline:

1. Creates deterministic, isolated synthetic adaptation and retention splits.
2. Trains a two-dimensional adapter for 20 CPU steps.
3. Evaluates task and representation metrics at four checkpoints.
4. Writes resumable checkpoints, a run manifest, checksummed summaries, and a compact web artifact.
5. Refuses to overwrite an existing run unless `--resume` is explicit.

Generated research outputs remain under ignored `artifacts/`. The reviewed smoke web derivative is written to `public/data/smoke.json`.

## Run tests

```bash
make test
```

The Python suite contains 41 tests covering configuration identity, deterministic and disjoint splits, revision-aware caching, duplicate-caption-safe supervision, task/geometry/layerwise metrics, rank-deficient covariance, gradient projection, post-hoc interpolation, weight-space ensembling, independent-run aggregation, held-out early warning, smoke execution, resume safety, and provenance. The web suite covers core loading, checkpoint synchronization, lazy detail exploration, view switching, and failure recovery.

## Commands

```bash
PYTHONPATH=src python -m driftlab smoke \
  --config configs/smoke.yaml \
  --resume \
  --web-output public/data/smoke.json

PYTHONPATH=src python -m driftlab validate-artifact public/data/smoke.json
```

Run the bounded local CLIP/Food-101/CIFAR-10 pipeline validation on the best available accelerator:

```bash
make reproduce-local
```

`configs/reproduction-local.yaml` records resolved dataset revisions and deterministic server selections and is labeled preliminary. Revision-addressed image bytes are cached under ignored `artifacts/cache/` for exact local reuse. A future portfolio-tier rerun must pin the recorded dataset revisions directly in configuration before execution.

Run the independent-seed baseline and mitigation laboratory:

```bash
make benchmark-local
make methods-local
make early-warning
```

Evaluate the three-seed post-hoc LoRA recovery curve:

```bash
PYTHONPATH=src .venv/bin/python -m driftlab interpolate-lora
```

Regenerate diagnostics byte-stably from saved model outputs without retraining:

```bash
PYTHONPATH=src .venv/bin/python -m driftlab benchmark-clip \
  --suite configs/reproduction-local-multiseed.yaml \
  --regenerate-metrics
```

## Portfolio application

```bash
cd apps/web
npm install
npm test
npm run build
```

The build verifies artifact schemas and manifest checksums before TypeScript/Vite compilation, then enforces gzip and data-size budgets. The main benchmark is loaded initially; the larger single-run class and embedding artifact is fetched only when a visitor opens the microscope. Route and standalone deployment options are documented in `apps/web/INTEGRATION.md`.

The project guide always has a deterministic, browser-only beginner mode. To enable grounded generative answers in a serverless deployment, copy `.env.example` to the deployment environment and set `OPENAI_API_KEY`; the key is read only by `api/project-guide.js` and never sent to the browser.

## Repository layout

- `src/driftlab/`: experiment and analysis package.
- `configs/`: versioned run definitions and research catalog.
- `tests/`: correctness and end-to-end smoke tests.
- `schemas/`: public artifact contracts.
- `data/legacy/`: external legacy-artifact registry; no course files are copied here.
- `reports/`: reconciliation and later research reports.
- `public/data/`: compact validated derivatives for the web experience.
- `apps/web/`: standalone, integration-ready React/Vite portfolio application.
- `services/inference/`: optional live inference service (future sprint).

## Legacy artifact verification

The original course archive is intentionally not committed. To verify registered files locally:

```bash
export DRIFTLAB_LEGACY_ROOT="/path/to/Deep Learning Final Project"
```

The manifest stores relative paths, byte counts, and SHA-256 values without embedding a private machine path.

```bash
PYTHONPATH=src python -m driftlab verify-legacy
```

Verify the separately supplied historical code scaffold:

```bash
export DRIFTLAB_LEGACY_CODE_ROOT="/path/to/DL Final"
PYTHONPATH=src python -m driftlab verify-legacy --manifest data/legacy/code-archive-manifest.json
```

## Scientific guardrails

- Observed correlation is not described as causation.
- Independent t-SNE fits are not treated as physical trajectories.
- Same-class repeated captions are positives in class-supervised contrastive training.
- New public runs must resolve to a configuration hash, dataset fingerprint, seed, model revision, committed code revision, environment snapshot, and artifact schema version. Existing pre-VCS local artifacts retain an explicit `uncommitted` provenance marker rather than being silently rewritten.
- Results with fewer than three valid seeds remain preliminary.
