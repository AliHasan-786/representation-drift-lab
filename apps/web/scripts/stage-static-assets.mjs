import { cp, mkdir, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repositoryPublic = resolve(here, "../../../public");
const stagedPublic = resolve(here, "../.public");

// `public/data/runs` contains full, reviewed per-seed diagnostic payloads.
// They remain versioned in the repository but are not fetched by the visitor
// experience. The app loads only the compact aggregate artifacts below; root
// manifests preserve its evidence-chip links.
const entries = [
  "data/benchmark-local.json",
  "data/benchmark-expanded-local.json",
  "data/early-warning-methodology.json",
  "data/method-comparison-local.json",
  "data/interpolation-local.json",
  "data/domain-comparison-local.json",
  "data/dataset-gallery.json",
  "data/reproduction-local.json",
  "data/manifests",
  "datasets",
  "report",
  "favicon.svg",
  "portfolio",
];

await rm(stagedPublic, { recursive: true, force: true });
await mkdir(stagedPublic, { recursive: true });
for (const entry of entries) {
  const source = resolve(repositoryPublic, entry);
  const destination = resolve(stagedPublic, entry);
  await mkdir(dirname(destination), { recursive: true });
  await cp(source, destination, { recursive: true });
}

console.log(`staged ${entries.length} offline-first asset groups without full run dumps`);
