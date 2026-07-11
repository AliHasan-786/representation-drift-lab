import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const publicRoot = resolve(here, "../../../public");

const readJson = async (relative) => {
  const source = await readFile(resolve(publicRoot, relative), "utf8");
  if (source.includes("/Users/") || source.includes("/home/")) {
    throw new Error(`${relative} contains a private absolute path`);
  }
  return JSON.parse(source);
};

const digest = async (path) =>
  createHash("sha256").update(await readFile(path)).digest("hex");

const validateSource = async (artifact) => {
  const path = resolve(publicRoot, artifact.source_manifest.public_path.replace(/^\//, ""));
  const actual = await digest(path);
  if (actual !== artifact.source_manifest.sha256) {
    throw new Error(`${artifact.run_id ?? artifact.artifact_id} manifest checksum mismatch`);
  }
};

const benchmark = await readJson("data/benchmark-local.json");
const detail = await readJson("data/reproduction-local.json");
const warning = await readJson("data/early-warning-methodology.json");
const methods = await readJson("data/method-comparison-local.json");
const interpolation = await readJson("data/interpolation-local.json");
const domains = await readJson("data/domain-comparison-local.json");
const gallery = await readJson("data/dataset-gallery.json");
const expanded = await readJson("data/benchmark-expanded-local.json");

for (const artifact of [benchmark, detail, warning, methods, interpolation, domains, expanded]) {
  if (artifact.schema_version !== "1.0.0") throw new Error("unsupported artifact schema");
  if (!/^[a-f0-9]{16}$/.test(artifact.config_hash)) throw new Error("invalid config hash");
  await validateSource(artifact);
}

const steps = benchmark.checkpoints.map((checkpoint) => checkpoint.step);
if (steps.join(",") !== [...new Set(steps)].sort((a, b) => a - b).join(",")) {
  throw new Error("benchmark checkpoints are not sorted and unique");
}
if (benchmark.experiment.run_count < 3 || benchmark.experiment.seeds.length < 3) {
  throw new Error("multi-seed benchmark must expose at least three runs");
}
for (const checkpoint of benchmark.checkpoints) {
  for (const role of ["retained", "adaptation"]) {
    const interval = checkpoint[role].top1_accuracy;
    if (interval.n !== benchmark.experiment.run_count || interval.ci_low > interval.mean || interval.ci_high < interval.mean) {
      throw new Error(`invalid ${role} uncertainty at step ${checkpoint.step}`);
    }
  }
}
if (expanded.experiment.run_count < 3 || expanded.experiment.seeds.length < 3 || expanded.evidence_status !== "local-multiseed-preliminary") {
  throw new Error("expanded benchmark lost its multi-seed preliminary status");
}
for (const run of expanded.runs) {
  await validateSource({ run_id: run.run_id, source_manifest: run.source_manifest });
}
if (expanded.checkpoints.at(-1).retained.accuracy_change.mean >= 0) {
  throw new Error("expanded benchmark no longer records the observed retained-score decline");
}
if (warning.evidence_status !== "synthetic-methodology-validation") {
  throw new Error("early-warning artifact lost its synthetic evidence label");
}
if (!warning.publication_caveat.includes("not evidence")) {
  throw new Error("early-warning artifact must retain its publication caveat");
}
if (methods.methods.length < 9 || methods.methods.some((method) => method.seeds.length < 3)) {
  throw new Error("method comparison must contain nine measured three-seed methods");
}
if (interpolation.curve.length !== 5 || interpolation.experiment.run_count < 3 || !interpolation.publication_caveat.includes("not an exact")) {
  throw new Error("interpolation artifact lost its recovery curve or fidelity caveat");
}
if (domains.scenarios.length !== 3 || domains.scenarios.some((scenario) => scenario.seeds.length < 3)) {
  throw new Error("domain comparison must contain three measured three-seed scenarios");
}
if (!domains.scenarios.some((scenario) => scenario.metrics.retained_accuracy_change.mean > 0)) {
  throw new Error("domain comparison lost its positive-transfer counterexample");
}
if (gallery.schema_version !== "1.0.0" || gallery.datasets.length !== 6 || gallery.datasets.some((dataset) => dataset.examples.length !== 4)) {
  throw new Error("dataset gallery must expose six datasets with four examples each");
}
for (const dataset of gallery.datasets) {
  for (const example of dataset.examples) {
    if (!example.src.startsWith(`/datasets/${dataset.id}/`) || !example.alt || !example.label) {
      throw new Error(`invalid dataset gallery example for ${dataset.id}`);
    }
    await readFile(resolve(publicRoot, example.src.replace(/^\//, "")));
  }
}
await readFile(resolve(publicRoot, "report/original-course-report.pdf"));

console.log(`validated ${benchmark.run_id}, ${expanded.run_id}, ${detail.run_id}, ${warning.artifact_id}, ${methods.run_id}, ${interpolation.run_id}, ${domains.run_id}, the dataset gallery, and the original report`);
