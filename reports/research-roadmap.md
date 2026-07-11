# Research Roadmap: From Local Demonstration to a Defensible Continual-Learning Study

## Purpose

Representation Drift Lab is intentionally honest about its current evidence: it is a locally reproducible, three-seed methodology demonstration—not a universal ranking of continual-learning methods. This document turns the next expansion into a sequence of falsifiable research gates. A method is not promoted to a portfolio conclusion merely because it is implemented or because one small run looks good.

## Non-negotiable rules

1. **No score-bearing image can influence an update, loss, checkpoint choice, or method selection.** Training, optional reference, validation, and final evaluation splits must be disjoint and their fingerprints published.
2. **Every comparison shares its score-bearing splits.** A method that receives extra reference images is labeled as a resource trade-off, not a free performance gain.
3. **A new method must beat at least one meaningful control on a preregistered primary metric while reporting its cost.** “Looks better on one chart” is not enough.
4. **A finding must survive at least three independent seeds and two domain scenarios before it is called a pattern.** Intervals and seed-level records remain public.
5. **Exploration and selection are separate.** Any parameter chosen after seeing a final score is reported as exploratory and gets a fresh confirmation run.

## Sprint A — strengthen the current benchmark before adding novelty

### Question

Does the local Food-101 → CIFAR-10 comparison still behave the same with enough held-out images to avoid the current ceiling and wide intervals?

### Design

- Increase the fixed, class-balanced Food-101 train/evaluation subsets and CIFAR-10 retained evaluation set while preserving all current disjointness checks.
- Add a validation split used only for choosing training duration, learning rate, and any retention weight. Keep the existing final retained set locked.
- Re-run the existing nine methods before admitting another method.
- Record wall-clock time, peak memory, and trainable parameters alongside adaptation accuracy, retained accuracy, zero-shot transfer, CKA, calibration, and confidence intervals.

### Promotion gate

Publish the expanded table only if every method completes three seeds, all manifests validate, and the frozen/zero-shot controls stay numerically reproducible. If the old ranking changes, publish the change rather than preserving the original narrative.

## Sprint B — an exact, fairer ZSCL-family control

### Why this is next

The current retention-distillation method is deliberately labeled *inspired*, not an exact reproduction. ZSCL studies zero-shot transfer degradation in continual CLIP training, uses a semantically diverse reference dataset for distillation, and combines feature-space and parameter-space protection. The current lab already has the hardest prerequisite: separate reference and final-evaluation images. [ZSCL (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Preventing_Zero-Shot_Transfer_Degradation_in_Continual_Learning_of_Vision-Language_Models_ICCV_2023_paper.html)

### Design

- Use a generic, semantically diverse image reference set that is disjoint from both the new task and every score-bearing retained set.
- Implement the paper’s relevant feature and parameter-space controls as separate ablations: reference distillation only, weight ensembling only, and their combination.
- Evaluate task adaptation, retained task accuracy, and a fixed suite of zero-shot domains. Do not use the zero-shot suite to choose hyperparameters.
- Compare against standard LoRA, current inspired distillation, and a frozen CLIP control at matched update budgets.

### Promotion gate

Call it an “adapted ZSCL reproduction” only after the implementation differences from the paper are enumerated, the validation protocol is fixed before final scoring, and the combined method’s value holds on at least two domain sequences.

## Sprint C — task-vector recovery and model merging

### Why this is next

The current adapter-output interpolation is a useful recovery curve, but it is not model merging. TIES-Merging was designed to reduce interference when combining independently fine-tuned task vectors by trimming small updates and resolving sign conflicts. [TIES-Merging](https://arxiv.org/abs/2306.01708)

### Design

- Train separate, independently initialized LoRA adapters for Food-101, EuroSAT, and Oxford Pets under matched protocols.
- Compare three post-hoc combinations: simple average, norm-scaled average, and a TIES-style sparse/sign-resolved merge.
- Score each source domain, each retained domain, and a held-out zero-shot suite. Report merge density, added storage, and whether task identity is required at inference.
- Hold out one task order for confirmation; do not select merge density on its final results.

### Promotion gate

The method earns a headline only if it improves a predeclared aggregate score over simple averaging without a statistically worse worst-domain score, and that claim is confirmed on a held-out task order.

## Sprint D — adaptive capacity, not just a single adapter

### Why this is next

Mixture-of-experts adapters offer a different trade-off: preserve the original CLIP path for unfamiliar data while routing in-distribution data to specialized adapters. This is promising for the project’s central stability/plasticity question, but it changes inference cost and routing assumptions. [MoE-Adapters for continual vision-language learning (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Boosting_Continual_Learning_of_Vision-Language_Models_via_Mixture-of-Experts_Adapters_CVPR_2024_paper.html)

### Design

- Add task-specialized adapters plus a documented router or a task-known routing baseline.
- Compare original-CLIP fallback, static task routing, and learned routing.
- Report routing accuracy, unknown-domain behavior, adapter count, latency, and memory—not only classification scores.
- Keep all training/reference/final splits disjoint for every task.

### Promotion gate

The extra complexity is justified only if a learned router beats the frozen fallback on known tasks without materially harming a preregistered unfamiliar-domain suite. A task-ID oracle is useful as an upper bound, never as a deployed claim.

## Research leads that remain exploratory

- **Continual LoRA with managed subspaces.** C-LoRA proposes routing and orthogonality to reuse low-rank capacity rather than growing a separate adapter per task. It is a useful lead, but this project should treat it as an unreplicated research direction until its assumptions are tested on the same protocol. [C-LoRA preprint](https://arxiv.org/abs/2502.17920)
- **Self-distillation and prompt adaptation.** Recent vision-language continual-learning work explores preserving transfer without a stored old-task dataset. This is attractive for privacy and storage, but requires a careful definition of what information the synthetic reference can encode before comparison.
- **Second model family.** Repeat the stable core protocol with a non-identical vision-language stack (for example, a modern SigLIP-family or OpenCLIP control) before interpreting a CLIP ViT-B/32 pattern as model-family behavior.

## Portfolio publication policy

The website should continue to distinguish four labels:

| Label | Meaning |
| --- | --- |
| Historical | Preserved course artifact; not silently upgraded by later work. |
| Local preliminary | Real, checksummed run; too small or narrow for a general method claim. |
| Confirmed comparison | Predeclared method comparison that passes the domain, seed, and split gates above. |
| Exploratory | Useful visual or hypothesis-generating result; not used to rank a method. |

This policy is a feature, not a disclaimer: the project is strongest when a visitor can see exactly what has been established, what has been falsified, and what remains an honest next question.
