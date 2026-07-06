import { gzipSync } from "node:zlib";
import { readdir, readFile, stat } from "node:fs/promises";
import { dirname, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dist = resolve(here, "../dist");

const collect = async (directory) => {
  const entries = await readdir(directory, { withFileTypes: true });
  const paths = [];
  for (const entry of entries) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) paths.push(...await collect(path));
    else paths.push(path);
  }
  return paths;
};

const files = await collect(dist);
const gzipTotal = async (extension) => {
  let total = 0;
  for (const path of files.filter((item) => extname(item) === extension)) {
    total += gzipSync(await readFile(path)).byteLength;
  }
  return total;
};

const budgets = [
  ["JavaScript gzip", await gzipTotal(".js"), 180 * 1024],
  ["CSS gzip", await gzipTotal(".css"), 35 * 1024],
  ["initial benchmark data", (await stat(resolve(dist, "data/benchmark-local.json"))).size, 250 * 1024],
  ["early-warning data", (await stat(resolve(dist, "data/early-warning-methodology.json"))).size, 20 * 1024],
  ["method comparison data", (await stat(resolve(dist, "data/method-comparison-local.json"))).size, 60 * 1024],
  ["interpolation data", (await stat(resolve(dist, "data/interpolation-local.json"))).size, 30 * 1024],
  ["domain comparison data", (await stat(resolve(dist, "data/domain-comparison-local.json"))).size, 30 * 1024],
  ["dataset gallery data", (await stat(resolve(dist, "data/dataset-gallery.json"))).size, 30 * 1024],
  ["dataset gallery images", (await Promise.all(files.filter((path) => path.includes("/datasets/")).map((path) => stat(path)))).reduce((sum, item) => sum + item.size, 0), 2 * 1024 * 1024],
  ["lazy detail data", (await stat(resolve(dist, "data/reproduction-local.json"))).size, 900 * 1024],
  ["original class report", (await stat(resolve(dist, "report/original-course-report.pdf"))).size, 2 * 1024 * 1024],
  ["extension report", (await stat(resolve(dist, "report/representation-drift-lab-report.pdf"))).size, 200 * 1024],
];

for (const [label, actual, limit] of budgets) {
  if (actual > limit) throw new Error(`${label} exceeded: ${actual} > ${limit} bytes`);
  console.log(`${label}: ${actual}/${limit} bytes`);
}
