# Portfolio integration contract

The application can ship independently or under an existing portfolio route.

## Option A — standalone deployment

Build `apps/web`, deploy `dist/`, and link to it from the portfolio project card. Preserve the `data/`, `datasets/`, and `report/` directories within the deployed app. The application is base-aware: to host it under a portfolio path, build with `VITE_BASE_PATH=/projects/representation-drift-lab/ npm run build` (or run `npm run build:subpath`). This prefixes public JSON, images, and PDFs automatically.

## Option B — route integration

Move `src/App.tsx`, `src/styles.css`, and the root `public/data` artifacts into the host React application. Mount the app at a dedicated route such as `/work/representation-drift-lab`. The host must serve these initial artifacts unchanged:

- `/data/benchmark-local.json`
- `/data/early-warning-methodology.json`
- `/data/method-comparison-local.json`
- `/data/interpolation-local.json`
- `/data/domain-comparison-local.json`
- `/data/dataset-gallery.json`
- `/datasets/*` (24 reviewed real examples across six datasets)
- `/report/original-course-report.pdf`
- `/report/representation-drift-lab-report.pdf`

The larger `/data/reproduction-local.json` artifact is deliberately loaded only after the visitor opens the class microscope.

## Optional grounded GenAI guide

The browser always includes deterministic beginner answers, so the project remains usable without a model provider. For generated answers, deploy `api/project-guide.js` as the same-origin `POST /api/project-guide` serverless route and set these server-only environment variables:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.4-mini
```

Do not rename the key with a `VITE_` prefix: that would expose it to client code. The endpoint sends no conversation history, caps questions and answers, disables provider-side response storage, and restricts answers to the project facts embedded in the server function. If the route or provider is unavailable, the UI automatically uses the offline guide.

## Required host behavior

- Do not cache HTML indefinitely; immutable JSON artifacts may use long-lived caching keyed by content hash at the CDN layer.
- Preserve the skip link, focus styles, reduced-motion behavior, headings, and text alternatives.
- Run `npm run validate:data`, `npm test`, and `npm run build` before deployment.
- Keep benchmark and live-inference results visually and semantically separate.
- Keep the original submitted report visually and semantically separate from the independent post-course extension.
- The static artifacts and all core interactions must continue to work during inference-service downtime.
