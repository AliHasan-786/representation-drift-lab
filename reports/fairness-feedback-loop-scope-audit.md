# Scope Audit: Fairness, Feedback Loops, and Human Impact

## Decision

This lab **does not make a fairness claim**. It evaluates representation
change and image-classification retention in bounded public benchmark tasks;
it is not a system that makes decisions about people, access, safety, or
services.

That boundary is deliberate. A clean experiment, a held-out test set, and a
three-seed interval are useful scientific checks. They are not evidence that
an AI system treats affected people fairly.

## What this project actually checks

| Check | Evidence | What it supports | What it does not support |
| --- | --- | --- | --- |
| Split isolation | Configuration fingerprints, disjointness tests, and manifests | Score-bearing images did not enter the recorded update or reference paths | Equitable outcomes for people or groups |
| Repetition | Three independent local seeds and Student-t intervals | Variation across controlled random starts in the stated benchmark | Reliability in a new population, context, or time period |
| Domain stress test | Food, satellite, pet, digit, and everyday-object pairs | The drift story can fail across selected image domains | Coverage of human identities, cultures, or high-stakes uses |
| Human release gate | Interactive, deliberately non-automatic review checklist | A release decision should have a named reviewer, scope, monitoring, and rollback plan | That any present model is safe to deploy |

## Why a demographic fairness score is not reported

The project datasets are labeled mainly by objects, food classes, locations,
pets, and digits. They do not supply a validated, purpose-appropriate set of
protected-attribute labels, affected-user outcomes, or a decision context.
Trying to calculate a demographic parity or error-rate gap from unsuitable
labels would manufacture precision rather than reveal harm. No such score is
shown on the site or used as a release gate.

## Feedback-loop assessment

There is no live model, user ranking, moderation workflow, or automated
decision in this project. Therefore there is no observed feedback loop to
measure. The interactive guide is a static/offline-first explanation tool;
it does not train the model from visitor inputs, and sensitive-looking guide
questions are blocked in the browser before any provider request.

If the adaptation method were proposed for a real workflow, the review could
not stop at benchmark metrics. Before any trial, an owner would need to
document:

1. The decision being assisted, who is affected, and which decisions remain
   human-owned.
2. Appropriate, consented evaluation slices and outcome measures, including
   error costs and an appeal or correction path.
3. How labels, overrides, complaints, or operator behavior could feed back
   into later model updates; no automatic learning from those signals by
   default.
4. Predeclared monitoring thresholds, a named incident owner, a rollback
   mechanism, and a way to pause the system when data distribution or harms
   change.

## Publication rule

Future work may add a fairness or feedback-loop result only when it states
the real decision context, the affected population, the provenance and
appropriateness of every slice label, the measured outcome, and the limits of
the audit. Until then, the correct status is **not assessed in this benchmark
setting**, not “passed.”
