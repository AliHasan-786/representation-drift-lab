# Full-scale benchmark: Kaggle free-tier runbook

This runbook executes the locked five-seed benchmark without paid compute. It
is an operational companion to
[`reports/full-scale-preregistration.md`](../reports/full-scale-preregistration.md),
not permission to alter its design.

## Locked experiment identity

- **Pinned revision:** `8966692d3121963ee288f8c695f2fd2ea3c8863a`
  (the immutable transport-compatible rerun revision printed by the notebook
  setup cell)
- **Configs:** `configs/full-scale-gpu.yaml` and
  `configs/full-scale-gpu-multiseed.yaml`
- **Seeds:** 201, 202, 203, 204, 205
- **Compute rule:** Kaggle free GPU only; no paid accelerator, no parameter
  changes, no configuration edits between seeds.

If a correction to the pinned code is needed, stop. Record the reason,
make a new preregistration/commit, and run the new identity; do not silently
patch a live benchmark session.

## Before opening Kaggle

1. Confirm the repository worktree is clean and the commit printed in the
   notebook setup cell is available on GitHub.
2. Read the preregistration once. The expected design is 32 Food-101 classes,
   16 training and 16 held-out evaluation images per class, a 500-image
   CIFAR-10 retained probe, 200 updates, and five seeds.
3. Use `kaggle/full-scale-benchmark.ipynb` from the pinned commit. Do not use a
   locally edited notebook after opening a session.

## Kaggle clicks (Ali signs in; Codex can guide after login)

1. Open [Kaggle Code](https://www.kaggle.com/code) and sign in.
2. Click **New Notebook**.
3. In the right sidebar, open **Notebook options** / **Settings** and set
   **Accelerator** to a free **GPU** option. Confirm the session shows a GPU;
   availability and GPU model are controlled by Kaggle.
4. Turn **Internet** on only if Kaggle requests it for the one-time GitHub and
   model/dataset downloads. The notebook does not need any credentials or
   secret tokens.
5. Upload `kaggle/full-scale-benchmark.ipynb` from the commit above, or paste
   its cells into the new notebook unchanged.
6. Run the setup cell. It must print the pinned commit before any benchmark
   command runs.
7. Run the GPU check. Stop if it reports `CUDA available: False`.
8. Run the suite cell once. Do not interrupt it to change hyperparameters. If
   Kaggle stops the session, reopen the same notebook and rerun the same suite
   cell; completed seeds are manifest-checked and skipped.
9. Run artifact validation. It must finish successfully before downloading any
   output.

## Required execution record

Before ending the session, save these facts in the notebook output or a local
note: Kaggle session date, reported GPU model, wall-clock start/end time, and
whether any resume occurred. Do not write result interpretation yet.

Download from the Kaggle output pane:

- `public/data/benchmark-full-scale-gpu.json`
- every matching file under `public/data/manifests/`
- the full console output or exported notebook

Keep all five seed manifests. A mean JSON file without the individual manifests
does not pass the project’s publication gate.

## Return-to-repository gate

1. Put the downloaded artifacts into their matching repository paths on a new
   branch; do not overwrite local preliminary artifacts.
2. Run `PYTHONPATH=src python -m driftlab validate-artifact
   public/data/benchmark-full-scale-gpu.json`.
3. Run the web artifact validator and build: `(cd apps/web && npm run build)`.
4. Update the preregistration only with execution facts (GPU, time, completion
   status). Publish the outcome whether it agrees with, weakens, or reverses
   the local story.

## Stop conditions

Stop rather than improvising if the GPU is unavailable, a dataset revision
cannot resolve, a seed fails validation, an artifact points to a different
commit/configuration, or the run requires payment. Preserve the error output
and resume only under the same locked identity.
