# Claude Fable 5 External Audit

## Audit provenance

- Auditor: Anthropic Claude Fable 5, invoked through Claude Code 1.0.35 with the explicit model identifier `claude-fable-5`.
- Date: 2026-07-06.
- Session: `9d817330-3bda-40ea-aad0-e606e06a6585`.
- Mode: read-only. Editing, writing, web access, and unrestricted shell access were disabled.
- Scope: specification, README, reports, manifests, experiment code, tests, public artifacts, React application, API endpoint, attribution, licensing, and deployment documentation.
- Limitation: the auditor could inspect test source but its validation commands were not permitted. Codex independently ran the Python, web, build, manifest, and PDF checks described below.
- Model availability was verified against Anthropic's official Fable 5 page: https://www.anthropic.com/claude/fable

## External verdict

Fable characterized the repository as unusually disciplined for a research portfolio, especially its provenance design, evidence labels, negative results, numerical safeguards, and beginner-oriented teaching. Its ship recommendation was conditional: correct the retained-evaluation leakage, establish Git history, and preserve the preliminary framing.

## Audit gates

| Gate | Fable verdict | Independent disposition |
| --- | --- | --- |
| Scientific claim support and leakage | Partial | Accepted. Two mitigation methods used the retained evaluation set as an optimization reference. A disjoint reference split was implemented and the affected suites must be rerun. |
| Reproducibility and provenance | Partial | Accepted. Git was initialized but had no commits. The new repository establishes history; pre-VCS artifacts retain their `uncommitted` marker instead of being rewritten. |
| Implementation correctness | Pass | Confirmed by 42 Python tests after remediation. |
| Security and privacy | Pass with minor gaps | Rate limiting and same-origin enforcement were added to the optional paid GenAI endpoint. |
| Accessibility and zero-assumption teaching | Pass with minor gaps | Arrow-key, Home, and End navigation plus tab/tabpanel linkage were added to interactive tab widgets. |
| Website, reports, and code consistency | Partial | Counts were normalized to the executable suite. A fourth invalid exploratory run is now explicitly documented as excluded. |
| Repository hygiene | Partial | A private GitHub repository was created. Temporary, environment, model, dataset, and build directories remain ignored. |
| Deployment readiness | Pass, not executed by Fable | Independently verified with the artifact validator, TypeScript, Vite production build, web tests, and bundle/data budgets. |

## Prioritized findings and response

### Accepted P1: retained-evaluation leakage

Fable correctly identified that `src/driftlab/clip_reproduction.py` assigned `datasets["retained_eval"]` to the optimization reference used by retention distillation and retention-gradient null-space projection. This violated the project's own isolation requirement and made those two retained-accuracy comparisons optimistic.

Response:

- The loader now supports a separate `retained_reference` range.
- Range overlap is rejected explicitly and covered by a unit test.
- Distillation and null-space configurations use CIFAR-10 rows 1000-1029 as the optimization reference while rows 0-29 remain evaluation-only.
- Baseline teacher logits for distillation are computed on the reference split, not indexed from evaluation outputs.
- A mitigation strategy now fails closed if no disjoint reference split is configured.
- The two affected three-seed suites and the combined method comparison must be regenerated before the corrected claims are published.

### Accepted P1: missing code history

The first Fable audit occurred before the repository had an initial commit. Existing manifests therefore honestly record `commit: "uncommitted"` and `dirty: true`.

Response:

- A private GitHub repository named `AliHasan-786/representation-drift-lab` was created.
- Existing pre-VCS artifacts are not relabeled retroactively; README guardrails now distinguish them from newly committed runs.
- Corrected runs generated after the initial commit record the committed source revision.

### Accepted P2: excluded fourth scenario

The repository contains a three-seed Pets-to-EuroSAT artifact that was not represented in the primary three-scenario comparison. Inspection shows 0% retained EuroSAT accuracy at baseline and final in every seed, so it cannot measure additional forgetting.

Response: `reports/excluded-runs.md` now preserves the run as a failed validity check, explains why it supports no scientific comparison, and states the gate for rerunning it.

### Accepted P2: paid endpoint abuse controls

The grounded project-guide endpoint limited request and response size but initially lacked traffic controls.

Response: the endpoint now enforces same-origin requests and an IP-scoped limit of ten requests per ten minutes, returns standard 403/429 responses, and keeps the existing server-only key, no-store policy, output cap, and offline fallback.

### Partially accepted P2: dataset revision pinning

Fable correctly observed that dataset revisions are resolved at runtime and recorded afterward rather than supplied in every YAML configuration. The cache provides exact local reuse but does not make a fresh upstream fetch immutable.

Response: the README no longer calls the selection intrinsically immutable and requires future portfolio-tier configurations to copy the resolved revisions into configuration before execution. Full revision-field enforcement remains a future migration because changing all current configuration identities would invalidate every existing run ID.

### Rejected P2: claimed 36-versus-38 test mismatch

Fable reported 36 tests based on static counting and could not execute them. Immediately before remediation, the executable suite reported 38 passing tests; subsequent disjoint-range, server-construction, and reference-split checks bring the executable suite to 41. The website and generated report now state 41.

### Accepted P3 items

- Completed EuroSAT, CIFAR-100, and Oxford Pets catalog entries were updated from `implementation-pending` to `local-benchmark-implemented`.
- Dataset, chart-decoder, and code-audit tabs now implement keyboard navigation and explicit `tabpanel` relationships.

### Retained backlog

- Pin dataset revisions inside all future portfolio-tier YAML files.
- Replace hard-coded chart domains and selected explanatory literals with artifact-derived bounds and values.
- Add a test double around the full CLIP strategy/resume loop.
- Clarify byte-stability as applying to derived metric records, not environment metadata across machines.
- Decide whether to wire continual-learning summary metrics into a future sequential-task benchmark or remove the unused production path.

## Independent validation

The following checks are the authoritative executable evidence for this sprint:

- `make test`: 42 Python tests.
- `npm test -- --run`: 6 web interaction tests.
- `npm run build`: schema/checksum validation, TypeScript, Vite build, and bundle/data budgets.
- `node --check api/project-guide.js`.
- Historical source manifest verification: 10 of 10 registered files valid.
- Extension PDF: all 10 pages rendered and visually inspected.

## Ship position

The site can ship as a preliminary research portfolio after the corrected distillation and null-space suites replace the leaked comparisons. The repository must continue to distinguish bounded protocol evidence from general scientific claims, and all future sprints must end with committed, pushed, reviewable state.
