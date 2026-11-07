import { cp, mkdir, readFile, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repositoryPublic = resolve(here, "../../../public");
const stagedPublic = resolve(here, "../.public");

// `public/data/runs` contains full, reviewed per-seed diagnostic payloads.
// They remain versioned in the repository but are not fetched by the visitor
// experience. The app loads the compact aggregate artifacts below. Every
// source manifest referenced by those aggregates is discovered and copied at
// its exact public path, including the few per-seed manifests that live under
// `data/runs/manifests`.
const aggregateArtifacts = [
  "data/benchmark-local.json",
  "data/benchmark-expanded-local.json",
  "data/early-warning-methodology.json",
  "data/method-comparison-local.json",
  "data/interpolation-local.json",
  "data/domain-comparison-local.json",
  "data/dataset-gallery.json",
  "data/reproduction-local.json",
];

const entries = [
  ...aggregateArtifacts,
  "datasets",
  "report",
  "favicon.svg",
  "portfolio",
];

const digest = (content) => createHash("sha256").update(content).digest("hex");

const sourceManifests = new Map();
const collectSourceManifests = (value) => {
  if (Array.isArray(value)) {
    value.forEach(collectSourceManifests);
    return;
  }
  if (!value || typeof value !== "object") return;
  const manifest = value.source_manifest;
  if (manifest?.public_path && manifest?.sha256) {
    sourceManifests.set(manifest.public_path, manifest.sha256);
  }
  Object.values(value).forEach(collectSourceManifests);
};

for (const artifact of aggregateArtifacts) {
  collectSourceManifests(JSON.parse(await readFile(resolve(repositoryPublic, artifact), "utf8")));
}

await rm(stagedPublic, { recursive: true, force: true });
await mkdir(stagedPublic, { recursive: true });
for (const entry of entries) {
  const source = resolve(repositoryPublic, entry);
  const destination = resolve(stagedPublic, entry);
  await mkdir(dirname(destination), { recursive: true });
  await cp(source, destination, { recursive: true });
}

for (const [publicPath, expectedDigest] of sourceManifests) {
  const relative = publicPath.replace(/^\//, "");
  let source = resolve(repositoryPublic, relative);
  let content;
  try {
    content = await readFile(source);
  } catch (error) {
    // Earlier local aggregate artifacts retained their historical public
    // manifest address while the source files now live beside the reviewed
    // per-seed run records. Preserve that public address only after checking
    // that this source byte-for-byte matches the aggregate's declared digest.
    if (!relative.startsWith("data/manifests/")) throw error;
    source = resolve(repositoryPublic, "data/runs/manifests", basename(relative));
    content = await readFile(source);
  }
  if (digest(content) !== expectedDigest) {
    throw new Error(`source manifest checksum mismatch before staging: ${publicPath}`);
  }
  const destination = resolve(stagedPublic, relative);
  await mkdir(dirname(destination), { recursive: true });
  await cp(source, destination);
}

console.log(`staged ${entries.length} offline-first asset groups and ${sourceManifests.size} checksummed manifests without full run dumps`);
