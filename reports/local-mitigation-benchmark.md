# Local Mitigation Benchmark

## Scope and publication status

This report documents a bounded methodology and engineering validation for the Representation Drift Lab. It compares nine real CLIP adaptation interventions across three independent seeds, but it is not a general method leaderboard. The subsets are intentionally small: each seed uses six Food-101 classes with 24 train and 24 evaluation images plus a 30-image CIFAR-10 retention probe.

Evidence status: **local-multimethod-preliminary**.

Canonical public derivative: `public/data/method-comparison-local.json`. Every row resolves to a source run, configuration hash, seed, model revision, dataset fingerprint, environment snapshot, saved-output checksums, and public manifest.

## Experimental controls

- Model: OpenAI CLIP ViT-B/32, resolved Hugging Face revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
- Adaptation: Food-101, resolved revision `83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9`.
- Retention: CIFAR-10, resolved revision `0b2714987fa478483af9968de7c934580d0bb9a2`.
- Seeds: 41, 42, and 43. Each method reuses the same seed-specific images and class selections.
- Training: 20 steps, batch size 4, learning rate 5e-4, checkpoints 0/5/10/20.
- Evaluation: top-1 accuracy, macro F1, NLL, ECE, baseline-relative change, fixed-baseline PCA Frechet distance, linear CKA, effective rank, local-neighborhood overlap, class-centroid movement, cross-modal alignment, and every vision-transformer block.
- Uncertainty: two-sided 95% Student-t intervals across three independent runs. Intervals are not clipped to metric bounds.

## Intervention definitions

| Method | Implementation | Fidelity label |
| --- | --- | --- |
| Frozen linear probe | Trains a 101-way linear head on frozen normalized CLIP image features; the encoder cannot drift | Exact frozen-encoder classification probe |
| Full fine-tune | Jointly trains the full vision encoder, projection, and a randomly initialized 101-way head for 200 steps | Full-parameter classification baseline |
| LP-FT | Trains the same head for 200 frozen-feature steps, then jointly tunes the full vision encoder and head for 20 steps | Adapted LP-FT classification baseline |
| Zero-shot initialized FT | Initializes the 101-way head from CLIP's scaled text prototypes, then jointly tunes the full vision model for 200 steps | Compatible WiSE-FT source baseline |
| WiSE-FT, alpha 0.5 | Interpolates the frozen and zero-shot-initialized fine-tuned vision model and compatible classifier weights | Adapted full-weight-space ensemble |
| Standard LoRA | Rank-8 LoRA on every vision-attention query and value projection; 294,912 trainable parameters | Corrected local pipeline validation |
| Retention distillation | Standard LoRA plus KL distillation from frozen baseline CIFAR-10 logits on a disjoint 30-image reference split | Inspired baseline, not an exact ZSCL reproduction |
| Gradient null-space | Projects each adaptation gradient tensor into the orthogonal complement of a gradient computed on the same disjoint retained reference split | Inspired first-order gradient-projection baseline |
| Selective LoRA | Rank-8 LoRA on value projections only; 147,456 trainable parameters | Selective parameter-efficient baseline |

The distillation design is motivated by [ZSCL](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Preventing_Zero-Shot_Transfer_Degradation_in_Continual_Learning_of_Vision-Language_Models_ICCV_2023_paper.html), but this small reference-replay implementation does not reproduce the full paper. The classification initialization experiment follows the feature-distortion motivation behind [LP-FT](https://arxiv.org/abs/2202.10054), and the compatible full-weight interpolation adapts [WiSE-FT](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html). Standard LoRA follows the parameter-efficient construction introduced by [Hu et al.](https://arxiv.org/abs/2106.09685). No row is labeled as an exact end-to-end paper reproduction.

## Results

Values are means with 95% intervals in brackets.

| Method | Final Food-101 accuracy | Food-101 change | Final CIFAR-10 accuracy | CIFAR-10 change | Retained 1 − CKA | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen linear probe | 1.0000 [1.0000, 1.0000] | +0.1806 [0.0610, 0.3001] | 0.7333 [0.7333, 0.7333] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 51,813 |
| Full fine-tune | 0.7917 [0.6124, 0.9709] | -0.0278 [-0.1859, 0.1303] | 0.7222 [0.5957, 0.8487] | -0.0111 [-0.1376, 0.1154] | 0.1011 [0.0904, 0.1117] | 87,901,029 |
| LP-FT | 1.0000 [1.0000, 1.0000] | +0.1806 [0.0610, 0.3001] | 0.7333 [0.7333, 0.7333] | 0.0000 [0.0000, 0.0000] | 0.00002 [-0.00001, 0.00004] | 87,901,029 |
| Zero-shot initialized FT | 0.9167 [0.7097, 1.1237] | +0.0972 [-0.1418, 0.3363] | 0.6000 [0.1390, 1.0610] | -0.1333 [-0.5944, 0.3277] | 0.1778 [0.0043, 0.3514] | 87,901,029 |
| WiSE-FT, alpha 0.5 | 0.9028 [0.7833, 1.0223] | +0.0833 [-0.1237, 0.2903] | 0.7667 [0.6839, 0.8495] | +0.0333 [-0.0495, 0.1161] | 0.0948 [-0.0385, 0.2281] | 87,901,029 |
| Standard LoRA | 0.8889 [0.8291, 0.9486] | +0.0694 [0.0097, 0.1292] | 0.7000 [0.6172, 0.7828] | -0.0333 [-0.1161, 0.0495] | 0.0126 [0.0105, 0.0147] | 294,912 |
| Retention distillation | 0.8750 [0.8750, 0.8750] | +0.0556 [-0.0640, 0.1751] | 0.7333 [0.7333, 0.7333] | 0.0000 [0.0000, 0.0000] | 0.0036 [0.0026, 0.0045] | 294,912 |
| Gradient null-space | 0.9167 [0.8132, 1.0202] | +0.0972 [-0.0609, 0.2553] | 0.6778 [0.6300, 0.7256] | -0.0556 [-0.1034, -0.0077] | 0.0097 [0.0060, 0.0135] | 294,912 |
| Selective LoRA | 0.9028 [0.8430, 0.9625] | +0.0833 [-0.0202, 0.1868] | 0.7111 [0.5027, 0.9195] | -0.0222 [-0.2306, 0.1862] | 0.0086 [0.0064, 0.0108] | 147,456 |

## What the local evidence says

1. The frozen linear probe perfectly separated this tiny evaluation subset while leaving the encoder and retained task unchanged. The score demonstrates separability and severe small-sample ceiling effects; it is not a credible full Food-101 estimate.
2. Full fine-tuning from a random 101-way head failed to exceed the zero-shot adaptation baseline after 200 joint steps and produced by far the largest representation change. This is an initialization and sparse-supervision failure case, not a universal indictment of full fine-tuning.
3. LP-FT inherited the probe's perfect tiny-subset separation and changed the encoder negligibly during its 20-step joint phase. This is consistent with the feature-distortion motivation for LP-FT, but the ceiling effect prevents a meaningful general method claim.
4. The zero-shot-initialized full fine-tune adapted well on average but had the worst mean retention and high seed variance. The adapted WiSE-FT ensemble recovered 16.7 retained-accuracy points relative to that source while preserving most of its adaptation mean.
5. On the corrected disjoint-reference rerun, retention distillation had the strongest mean retained accuracy among LoRA encoder-changing methods, while giving up 1.4 adaptation-accuracy points versus standard LoRA. It is a resource trade-off: it receives retained reference images that plain LoRA does not.
6. Gradient null-space adaptation had the highest mean adapted accuracy among LoRA methods and lower mean CKA loss than standard LoRA, yet the worst mean retained accuracy of those methods.
7. Selective LoRA cut trainable parameters in half and produced the lowest nonzero mean CKA loss among LoRA methods, but its retained-accuracy interval was the widest.
8. Geometric drift did not reliably rank methods by retained performance. Across nine intervention means, CKA loss versus mean forgetting had Pearson r = 0.5226, Spearman rho = 0.4577, and a 95% paired-bootstrap Pearson interval of [-0.7646, 0.9635]. This is a negative diagnostic—not evidence of a stable relationship.

## Failure cases worth publishing

- **Drift without mean forgetting:** retention distillation returned mean retained accuracy to baseline while retained representations still moved, even after the reference data was separated from evaluation data.
- **Lower drift with worse retention:** gradient null-space adaptation reduced mean CKA loss relative to standard LoRA but had lower mean retained accuracy.
- **Metric disagreement:** selective LoRA ranked best on mean CKA preservation but not on retained accuracy.

These cases are why the application exposes raw task metrics and multiple geometric measures instead of collapsing them into one “safety” score.

## Post-hoc recovery curve

A separate three-seed analysis scales the trained standard-LoRA output from alpha 0 (frozen model) to alpha 1 (full adapter). This is an exact scaling of the adapter's function-space update, but only a **WiSE-FT-inspired** recovery experiment because it does not interpolate two full fine-tuned model states.

| Adapter scale | Mean retained accuracy | Mean adapted accuracy | Mean retained 1 − CKA |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.7333 | 0.8194 | 0.0000 |
| 0.25 | 0.7111 | 0.8472 | 0.0008 |
| 0.50 | 0.7222 | 0.8889 | 0.0031 |
| 0.75 | 0.7111 | 0.9028 | 0.0070 |
| 1.00 | 0.7000 | 0.8889 | 0.0126 |

Alpha 0.5 matched the full adapter's mean adapted accuracy while recovering 2.2 retained-accuracy points. Alpha 0.75 exceeded the full adapter's mean adapted accuracy while recovering 1.1 retained points. These are inspected points on the same evaluation data, not a held-out hyperparameter-selection result.

## Compute and reproducibility

Runs used PyTorch 2.11 on Apple Metal Performance Shaders (`arm64`, macOS/Darwin). The full environment records also pin Transformers 5.13.0, PEFT 0.19.1, Datasets 5.0.0, Accelerate 1.14.0, Pillow 12.3.0, and the exact Python/numerical stack. Network retrieval and cached preprocessing are separated from run artifacts; revision-addressed dataset rows and image bytes are reused across methods.

Historical timing fields for the first local runs begin after model/data initialization and therefore are not presented as end-to-end compute measurements. New runs start timing at invocation. Training cost in the current public matrix is consequently reported as recorded steps and trainable parameters, not an unreliable wall-clock comparison.

## Threats to validity

- Three seeds are the minimum uncertainty gate, not a large sample.
- The class-balanced Food-101 subset and tiny CIFAR-10 probe are useful for pipeline testing but underpowered for scientific ranking.
- All methods use one CLIP backbone, one adaptation domain, one retained domain, and one short high-learning-rate schedule.
- Seed-specific datasets differ by design, increasing ecological variation but also widening intervals.
- Distillation has access to retained reference examples; standard and selective LoRA do not. This is a resource trade-off, not a free improvement.
- The null-space projection is tensorwise and first-order; it does not guarantee global functional invariance.
- CKA, centroid drift, and accuracy measure different objects. Agreement should not be assumed.

## Next experimental gates

Before promoting a method ranking to a primary portfolio claim, the benchmark must run larger samples, cover additional adaptation and retained domains, include at least two additional model/control families, and test the paper protocols under realistic schedules. Adaptive-rank LoRA and MergeTune-style recovery remain method-expansion targets.
