# Representation Drift Lab — portfolio application

This standalone Vite application is integration-ready for Ali Hasan's portfolio. It reads versioned artifacts from the repository-level `public/data` directory and requires no inference service for its core experience.

```bash
npm install
npm run dev
npm run test
npm run build
```

The production build runs artifact-contract checks before compilation and enforces JavaScript and data performance budgets afterward. See `INTEGRATION.md` for route and link integration options.
