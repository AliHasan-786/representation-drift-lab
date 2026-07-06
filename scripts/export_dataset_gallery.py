from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "artifacts/cache/huggingface/rows"
IMAGES = ROOT / "artifacts/cache/huggingface/images"
OUTPUT = ROOT / "public/datasets"
MANIFEST = ROOT / "public/data/dataset-gallery.json"

DATASETS = [
    {
        "id": "food101",
        "repository": "ethz/food101",
        "name": "Food-101",
        "plain_name": "photographs of prepared dishes",
        "label_field": "label",
        "full_size": "101,000 images in 101 food categories",
        "native_resolution": "varied photograph sizes",
        "role": "Learning dataset",
        "role_explanation": "The model updates itself using food images so it becomes more specialized at recognizing dishes.",
        "local_protocol": "Each local seed uses 6 classes, with 4 training and 4 evaluation images per class.",
        "source_url": "https://huggingface.co/datasets/ethz/food101",
    },
    {
        "id": "cifar10",
        "repository": "uoft-cs/cifar10",
        "name": "CIFAR-10",
        "plain_name": "tiny photographs of everyday objects and animals",
        "label_field": "label",
        "full_size": "60,000 images in 10 categories",
        "native_resolution": "32 x 32 pixels",
        "role": "Memory check",
        "role_explanation": "The model never trains on these examples. They test whether learning food harms an older general object-recognition ability.",
        "local_protocol": "Each local seed evaluates the same type of 30-image retained probe.",
        "source_url": "https://huggingface.co/datasets/uoft-cs/cifar10",
    },
    {
        "id": "eurosat",
        "repository": "blanchon/EuroSat",
        "name": "EuroSAT",
        "plain_name": "satellite images of land use",
        "label_field": "label",
        "full_size": "27,000 images in 10 land-use categories",
        "native_resolution": "64 x 64 pixels",
        "role": "Learning dataset",
        "role_explanation": "This second scenario asks the same model to specialize in overhead satellite imagery.",
        "local_protocol": "Each local seed uses 6 classes, with 4 training and 4 evaluation images per class.",
        "source_url": "https://huggingface.co/datasets/blanchon/EuroSat",
    },
    {
        "id": "cifar100",
        "repository": "uoft-cs/cifar100",
        "name": "CIFAR-100",
        "plain_name": "tiny photographs from many fine-grained categories",
        "label_field": "fine_label",
        "full_size": "60,000 images in 100 categories",
        "native_resolution": "32 x 32 pixels",
        "role": "Memory check",
        "role_explanation": "These images test whether satellite specialization disrupts a broader set of object categories.",
        "local_protocol": "Each local seed evaluates a 40-image retained probe.",
        "source_url": "https://huggingface.co/datasets/uoft-cs/cifar100",
    },
    {
        "id": "pets",
        "repository": "timm/oxford-iiit-pet",
        "name": "Oxford-IIIT Pet",
        "plain_name": "cat and dog breed photographs",
        "label_field": "label",
        "full_size": "7,349 images across 37 pet breeds",
        "native_resolution": "varied photograph sizes",
        "role": "Learning dataset",
        "role_explanation": "This third scenario specializes the model toward distinguishing visually similar cat and dog breeds.",
        "local_protocol": "Each local seed uses 6 breeds, with 3 training and 2 evaluation images per breed.",
        "source_url": "https://huggingface.co/datasets/timm/oxford-iiit-pet",
    },
    {
        "id": "mnist",
        "repository": "ylecun/mnist",
        "name": "MNIST",
        "plain_name": "handwritten digits from zero through nine",
        "label_field": "label",
        "full_size": "70,000 handwritten digit images in 10 categories",
        "native_resolution": "28 x 28 grayscale pixels",
        "role": "Memory check",
        "role_explanation": "Digits are visually very different from pets, making this a useful counterexample: retained performance improved even though representations moved.",
        "local_protocol": "Each local seed evaluates a 40-image retained probe.",
        "source_url": "https://huggingface.co/datasets/ylecun/mnist",
    },
]


def image_source(row: dict) -> str | None:
    for value in row.values():
        if isinstance(value, dict) and "src" in value:
            return str(value["src"])
    return None


def label_names(payload: dict, field: str) -> list[str]:
    for feature in payload["features"]:
        if feature["name"] == field:
            return [str(name).replace("_", " ") for name in feature["type"]["names"]]
    raise ValueError(f"Missing label names for {field}")


def export_image(source: str, destination: Path) -> None:
    cache = IMAGES / f"{hashlib.sha256(source.encode()).hexdigest()}.bin"
    if not cache.exists():
        raise FileNotFoundError(f"Cached experiment image is missing: {cache.name}")
    with Image.open(BytesIO(cache.read_bytes())) as raw:
        image = raw.convert("RGB")
        resample = Image.Resampling.NEAREST if min(image.size) <= 64 else Image.Resampling.LANCZOS
        fitted = ImageOps.fit(image, (640, 480), method=resample)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fitted.save(destination, "WEBP", quality=82, method=6)


def main() -> None:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(ROWS.glob("*.json"))]
    manifest = {"schema_version": "1.0.0", "datasets": []}
    for spec in DATASETS:
        selected: list[dict] = []
        seen_labels: set[int] = set()
        for payload in payloads:
            for item in payload.get("rows", []):
                row = item["row"]
                source = image_source(row)
                if not source or f"cached-assets/{spec['repository']}/" not in source:
                    continue
                label = int(row[spec["label_field"]])
                if label in seen_labels:
                    continue
                cache = IMAGES / f"{hashlib.sha256(source.encode()).hexdigest()}.bin"
                if not cache.exists():
                    continue
                names = label_names(payload, spec["label_field"])
                destination = OUTPUT / spec["id"] / f"{len(selected) + 1}-{label}.webp"
                export_image(source, destination)
                selected.append(
                    {
                        "src": f"/datasets/{spec['id']}/{destination.name}",
                        "label": names[label],
                        "row_index": int(item["row_idx"]),
                        "alt": f"A real {spec['name']} example labeled {names[label]}",
                    }
                )
                seen_labels.add(label)
                if len(selected) == 4:
                    break
            if len(selected) == 4:
                break
        if len(selected) < 4:
            raise RuntimeError(f"Only found {len(selected)} cached examples for {spec['name']}")
        manifest["datasets"].append({key: value for key, value in spec.items() if key != "repository"} | {"examples": selected})
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    print(f"Exported {sum(len(item['examples']) for item in manifest['datasets'])} real experiment images")


if __name__ == "__main__":
    main()
