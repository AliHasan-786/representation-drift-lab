# Vercel deployment handoff

This project deploys from the repository root. The checked-in
[`vercel.json`](../vercel.json) installs and builds the Vite application in
`apps/web`, serves `apps/web/dist`, and leaves the root-level `api/` directory
available for the optional same-origin project-guide function.

## Before connecting Vercel

Run this from the repository root:

```bash
make test
make api-test
(cd apps/web && npm test -- --run && npm run build)
```

Do not add `OPENAI_API_KEY` for the first deployment. The site is complete
without it: the guide falls back to its local, curated answers. Adding a model
provider key turns on a paid/usage-metered external service and should be a
separate, deliberate decision.

## Account-bound steps (Ali)

1. Go to [Vercel’s New Project page](https://vercel.com/new) and sign in with
   the GitHub account that can access `AliHasan-786/representation-drift-lab`.
2. Under **Import Git Repository**, select `representation-drift-lab` and click
   **Import**.
3. In **Configure Project**, leave **Root Directory** as `.`. The repository
   config already sets the install command, build command, and output
   directory; do not replace them in the dashboard.
4. Leave **Environment Variables** empty for this first, offline-first deploy.
5. Click **Deploy**. This is the point at which Vercel creates the account-owned
   project and deployment; it cannot be done safely without Ali’s signed-in
   account.
6. After Vercel reports success, open the deployment URL and test:
   - the original report PDF;
   - one dataset image gallery;
   - the `Load expanded confirmation` button;
   - an evidence-tier chip (it should open JSON); and
   - the project guide with a normal question and a fake email address (the
     latter should remain in the browser).

## Optional custom domain / portfolio route

Use the standalone Vercel URL first. If the site later moves under an existing
portfolio path, rebuild with `VITE_BASE_PATH=/projects/representation-drift-lab/`
as documented in `apps/web/INTEGRATION.md`; do not configure a subpath only in
the Vercel dashboard.

## Optional generated guide (separate decision)

Only after the static deployment is verified, add `OPENAI_API_KEY` and
`OPENAI_MODEL` under **Project → Settings → Environment Variables**, redeploy,
and test the same-origin `POST /api/project-guide` route. The key must never be
named with a `VITE_` prefix or committed to the repository.
