# Results: Representation Drift Lab

## Headline

In a bounded, three-seed local CLIP study, adapting a model to a new image task
could improve the new task while changing retained-task performance and internal
representations in different ways. No single representation-drift metric reliably
ranked retained performance across the nine local interventions.

This is preliminary local evidence, not a general method leaderboard, a
deployment threshold, a forecast of real forgetting, or a claim about fairness
for people.

## What was measured

- **Model and task:** CLIP ViT-B/32 adapted from Food-101 to a CIFAR-10
  retention probe using rank-8 LoRA adapters.
- **Local benchmark:** three independent seeds, six Food-101 classes with 24
  training and 24 held-out evaluation images per seed, and a 30-image CIFAR-10
  retained probe. The compact aggregate is
  [`public/data/benchmark-local.json`](public/data/benchmark-local.json).
- **Method comparison:** nine interventions across three seeds, with two-sided
  95% Student-t intervals. The evidence table is
  [`reports/local-mitigation-benchmark.md`](reports/local-mitigation-benchmark.md)
  and the machine-readable artifact is
  [`public/data/method-comparison-local.json`](public/data/method-comparison-local.json).
- **Expanded confirmation:** a preregistered local check used eight Food-101
  classes, 8 train + 8 held-out evaluation images per class, a 100-image CIFAR-10
  probe, 50 updates, and seeds 101–103. Its artifact is
  [`public/data/benchmark-expanded-local.json`](public/data/benchmark-expanded-local.json).

## Observed local patterns

1. **Adaptation and retention can trade off.** Standard LoRA had mean final
   Food-101 accuracy of 88.9% and mean retained CIFAR-10 accuracy of 70.0%, a
   -3.3 percentage-point retained change versus its local baseline. Its retained
   CKA loss was 0.0126. These are three-seed local estimates with wide,
   unclipped intervals—not population estimates.
2. **Geometry and retained accuracy disagreed.** Across nine intervention
   means, retained CKA loss versus mean forgetting had Pearson r = 0.5226 with
   a paired-bootstrap 95% interval of [-0.7646, 0.9635]. That interval includes
   many relationships, so the result is a negative diagnostic rather than
   evidence of a stable predictive relationship.
3. **Mitigations expose trade-offs rather than a winner.** Retention
   distillation returned mean retained accuracy to baseline in the local setup,
   but it uses retained reference images that standard LoRA does not. Selective
   LoRA halved trainable parameters and had the lowest nonzero mean CKA loss
   among LoRA variants, but its retained-accuracy interval was widest.
4. **An expanded local check reproduced the directional trade-off.** It is still
   one backbone, one domain pair, and three seeds; it does not establish a broad
   comparative ranking.

## What is not a result

- The submitted 2025 course report is preserved as historical coursework and is
  not silently relabeled as independent confirmatory evidence.
- The early-warning evaluator currently uses synthetic predictor observations.
  It validates evaluation mechanics, not the ability to forecast real
  forgetting.
- The Pets-to-EuroSAT exploratory pair started at a 0% retained-score floor in
  all three seeds and is preserved but excluded from the three-scenario
  comparison.
- The five-seed full-scale GPU tier has **not completed**. No aggregate artifact
  or five validated seed manifests have been published.

## Next evidence gate

The preregistered full-scale tier is a separate milestone: 32 Food-101 classes,
512 train and 512 held-out evaluation images, a 500-image CIFAR-10 retention
probe, 200 LoRA updates, and seeds 201–205. It can be promoted only when all
five manifests validate and the frozen/zero-shot controls reproduce. The first
Kaggle attempt stopped before seed 201 due to an external dataset-server request
limit; the transport-compatible rerun code is pinned at
`8966692d3121963ee288f8c695f2fd2ea3c8863a`.

See [`reports/full-scale-preregistration.md`](reports/full-scale-preregistration.md)
for the locked protocol and [`reports/research-roadmap.md`](reports/research-roadmap.md)
for the broader evidence plan.
