# Historical Source-Code Audit

## Scope and verdict

The folder `/Users/alihasan/Downloads/DL Final` adds a historical source-code scaffold that was not present in the previously audited course directory. It contains 42 files at the outer-folder level, including a 46,659-byte ZIP, an extracted `drift-analysis` tree, an empty `test.py`, and an outer `.gitignore`.

This is useful lineage evidence, but it is not new experimental evidence. The included README explicitly says that datasets, checkpoints, embeddings, results, and logs are absent. Both notebooks contain zero executed cells and zero outputs. No number from this folder is promoted to a measured result.

## Identity and duplication check

- Archive SHA-256: `957682e7d364010b8d05f452eca1e141967daf00b53eb58c2946ca3bdb5ff645`.
- Extracted-tree digest: `3fcbf170707783e446b9f7738507d918ff20640c403d68e473009d716cb97ef4`, computed from the sorted per-file SHA-256 listing rooted at `drift-analysis`.
- The ZIP and extracted tree contain the same source files except that the extracted README is a later, expanded version and the extracted tree has an empty `data/datasets` directory.
- The later README candidly identifies seven placeholder modules and states that the data and run outputs are not included.
- The checksummed external registry is `data/legacy/code-archive-manifest.json`.

## What the source establishes

The code records an intended research direction:

- Start from `openai/clip-vit-base-patch32`.
- Add rank-8 LoRA adapters with alpha 16 and dropout 0.1 to vision-attention `q_proj` and `v_proj` modules.
- Fine-tune on Food-101 using a symmetric image-text contrastive loss.
- Evaluate image/text embeddings on COCO and potentially Flickr30k.
- Compute per-sample cosine change, centroid displacement, Frechet-style distribution distance, image-text alignment, semantic-cluster drift, velocity, and acceleration.
- Explore additional loaders for ImageNet, Visual Genome, ChestXray14, and UCF101.
- Generate static and Plotly visualizations, including joint t-SNE/UMAP comparisons and checkpoint animations.

These are design intentions visible in source. The folder does not prove that the complete path executed or produced the report's measurements.

## Why it cannot be the complete producing pipeline

The scaffold and the submitted study diverge in important ways:

1. The primary scripted retention dataset is COCO; this source contains no CIFAR-10 loader, while the submitted report and traced notebook use CIFAR-10.
2. The pipeline expects step-level embedding files, but the fine-tuner extracts embeddings only after epochs.
3. The two notebooks are templates with no execution counts or stored outputs.
4. Seven modules are explicit placeholders: forgetting evaluation, downstream evaluation, predictive modeling, experiment comparison, result tables, medical-data preparation, and paper-figure generation.
5. The source archive contains no configuration snapshot, model revision, dataset fingerprint, seed record, checkpoint, metric CSV, or log from a completed run.

The correct conclusion is that this is an earlier or parallel implementation scaffold, not the missing producer for legacy evidence Families A or B.

## Reproducibility and correctness findings

All 32 Python files parse successfully, and all three shell scripts pass shell syntax checking. Static review nevertheless found execution blockers and scientific risks:

| Finding | Consequence | Extension response |
| --- | --- | --- |
| `run_complete_pipeline.sh` calls a plural `--datasets` option, but the downloader defines only singular `--dataset`; it also requests an unregistered `medical` key. | The advertised full pipeline stops during data setup. | Versioned dataset-server selections and revision-aware caches replace the ad hoc downloader. |
| The same script passes `--eval_samples` to `finetune_lora.py`, whose parser does not define that option. | Fine-tuning exits before training. | Every public configuration is parsed and exercised by automated tests. |
| Data tests import `CachedCOCODataset` and `CachedFood101Dataset`, which do not exist in the loader module. | The historical test suite cannot collect as written. | The extension currently has 38 passing Python checks, including end-to-end smoke and cache behavior. |
| Checkpoints save PEFT/LoRA state, but `load_finetuned_clip` reconstructs only plain CLIP before strict state loading. | Standalone checkpoint extraction cannot reliably restore the adapter topology. | The extension reconstructs the exact strategy before loading and validates trainable-parameter invariants. |
| Food-101 captions repeat within a batch, but the diagonal-only contrastive loss treats same-class captions as negatives. | Correct same-class pairs can be penalized as false negatives. | The rebuilt loss uses class-aware multi-positive supervision with a dedicated test. |
| `num_samples` is ignored by the Food-101 loader, while COCO expands each selected image into multiple caption rows. | Command-line sample counts do not consistently describe actual examples. | Split identities, selected rows, counts, and fingerprints are recorded in each manifest. |
| “Catastrophic forgetting” is inferred from a drift threshold without requiring an observed performance decline. | Internal change is mislabeled as capability loss. | Drift and task forgetting are measured and reported separately. |
| The early predictor can use velocities from later checkpoints on the same trajectory. | The apparent forecast can leak future information. | The extension separates train/validation/test scenarios and labels its current predictor artifact synthetic. |
| Checkpoint order is inferred from lexicographically sorted filenames and epoch steps are guessed as `epoch * 5000`. | Time order and step counts can be wrong. | Saved checkpoints carry explicit steps and are schema-validated. |

## What is genuinely new for the portfolio story

The folder adds evidence of the project's engineering evolution. It shows the original ambition—multi-domain loaders, multimodal geometry, checkpoint trajectories, semantic clusters, and an interactive Plotly dashboard—while also making the incompleteness inspectable. The portfolio should present this as code archaeology:

1. The course project generated a strong question.
2. Several code and artifact branches evolved separately.
3. A source scaffold alone did not make the claims reproducible.
4. The independent extension converted the useful ideas into a tested, traceable system and rejected unsupported conclusions.

This history strengthens the project only when it is explicit. It demonstrates audit discipline and scientific correction; it must not be used to imply that the scaffold's unexecuted medical, video, predictive, or downstream plans were completed.
