# Expanded Local Validation: Preregistered Protocol

## Why this run exists

The published Food-101 → CIFAR-10 local benchmark is a reproducibility and methodology check. It deliberately uses six Food-101 classes, four train and four evaluation examples per class, 20 training updates, and a 30-image retained probe. Those choices are fast enough to inspect deeply, but too small for a stronger comparative claim.

This expanded run increases the fixed local evidence without changing the model family or selectively tuning after a result is seen.

## Fixed before execution

| Decision | Fixed value |
| --- | --- |
| Model | `openai/clip-vit-base-patch32` |
| Adapter | rank-8 LoRA, alpha 16, query/value projections |
| Seeds | 101, 102, 103 |
| Adaptation data | 8 Food-101 classes; 8 train + 8 held-out evaluation images per class and seed, selected from a fixed 96-row dataset-server window |
| Retained evaluation | 100 CIFAR-10 test images starting at row 200; no retained reference images are used |
| Schedule | 50 updates; checkpoints 0, 10, 25, 50 |
| Batch sizes | 8 training / 16 evaluation |
| Learning rate | 5e-4 |
| Geometry projection | fixed baseline PCA, 32 dimensions |

The exact executable definitions are `configs/reproduction-expanded-local.yaml` and `configs/reproduction-expanded-local-multiseed.yaml`. Their hashes, selected rows, model revision, environment, and output checksums will be published with the run.

## Provenance correction before publication

An initial execution created output files before its manifest captured Git status, causing the manifest to report a dirty tree even though the protocol was committed beforehand. That artifact is retained locally for debugging but is not used as public evidence. Before the publishable rerun, provenance capture was moved to invocation start and generated-output paths were excluded from the source-dirty check. The rerun keeps every scientific setting above unchanged; only its output directory changes so it cannot overwrite the diagnostic artifact.

## Questions and interpretation limits

This run asks whether the direction of the original local story—new-task adaptation, retained-task change, and representational movement—survives a larger fixed local sample. It does **not** establish a broad model ranking, validate a deployment threshold, or test a second model family. Any comparison remains `expanded-local-preliminary` until the research-roadmap domain and confirmation gates are met.
