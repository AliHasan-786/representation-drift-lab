from __future__ import annotations

import hashlib
import itertools
import json
import time
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageDataset:
    name: str
    split: str
    revision: str
    selection: dict[str, Any]
    images: tuple[Any, ...]
    labels: tuple[int, ...]
    class_names: tuple[str, ...]
    fingerprint: str


def _selection_fingerprint(
    *,
    repository: str,
    split: str,
    revision: str,
    selection: dict[str, Any],
    labels: tuple[int, ...],
) -> str:
    payload = {
        "repository": repository,
        "split": split,
        "revision": revision,
        "selection": selection,
        "labels": labels,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def resolve_dataset_revision(repository: str) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "research dependencies are required; install requirements.research.lock"
        ) from error
    return str(HfApi().dataset_info(repository).sha)


def _materialize_streaming_range(dataset: Any, start: int, size: int) -> list[dict[str, Any]]:
    if start < 0 or size < 1:
        raise ValueError("streaming selection start/size are invalid")
    return list(itertools.islice(iter(dataset), start, start + size))


def _from_records(
    *,
    repository: str,
    split: str,
    revision: str,
    image_field: str,
    label_field: str,
    records: list[dict[str, Any]],
    class_names: tuple[str, ...],
    selection: dict[str, Any],
) -> ImageDataset:
    labels = tuple(int(record[label_field]) for record in records)
    images = tuple(record[image_field].convert("RGB") for record in records)
    if not images:
        raise ValueError("dataset selection is empty")
    return ImageDataset(
        name=repository,
        split=split,
        revision=revision,
        selection=selection,
        images=images,
        labels=labels,
        class_names=class_names,
        fingerprint=_selection_fingerprint(
            repository=repository,
            split=split,
            revision=revision,
            selection=selection,
            labels=labels,
        ),
    )


def load_streaming_range(
    spec: dict[str, Any], *, start: int, size: int, revision: str | None = None
) -> ImageDataset:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "research dependencies are required; install requirements.research.lock"
        ) from error
    repository = str(spec["repository"])
    split = str(spec["split"])
    resolved = revision or resolve_dataset_revision(repository)
    dataset = load_dataset(repository, split=split, streaming=True, revision=resolved)
    label_field = str(spec.get("label_field", "label"))
    class_names = tuple(dataset.features[label_field].names)
    records = _materialize_streaming_range(dataset, start, size)
    return _from_records(
        repository=repository,
        split=split,
        revision=resolved,
        image_field=str(spec["image_field"]),
        label_field=label_field,
        records=records,
        class_names=class_names,
        selection={"mode": "streaming-prefix", "start": start, "size": size},
    )


def load_materialized_seeded_splits(
    *,
    adaptation_spec: dict[str, Any],
    retained_spec: dict[str, Any],
    seed: int,
) -> dict[str, ImageDataset]:
    try:
        import numpy as np
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "research dependencies are required; install requirements.research.lock"
        ) from error
    results = {}
    for role, spec in (("adaptation", adaptation_spec), ("retained", retained_spec)):
        repository = str(spec["repository"])
        split = str(spec["split"])
        revision = resolve_dataset_revision(repository)
        dataset = load_dataset(repository, split=split, revision=revision)
        label_field = str(spec.get("label_field", "label"))
        class_names = tuple(dataset.features[label_field].names)
        permutation = np.random.default_rng(seed + (0 if role == "adaptation" else 1000)).permutation(len(dataset))
        if role == "adaptation":
            eval_size = int(spec["eval_size"])
            train_size = int(spec["train_size"])
            eval_indices = permutation[:eval_size].tolist()
            train_indices = permutation[eval_size : eval_size + train_size].tolist()
            for suffix, indices in (("eval", eval_indices), ("train", train_indices)):
                records = [dataset[int(index)] for index in indices]
                results[f"adaptation_{suffix}"] = _from_records(
                    repository=repository,
                    split=split,
                    revision=revision,
                    image_field=str(spec["image_field"]),
                    label_field=label_field,
                    records=records,
                    class_names=class_names,
                    selection={
                        "mode": "materialized-seeded-split",
                        "seed": seed,
                        "indices_sha256": hashlib.sha256(
                            json.dumps(indices).encode("utf-8")
                        ).hexdigest(),
                        "size": len(indices),
                    },
                )
        else:
            indices = permutation[: int(spec["eval_size"])].tolist()
            records = [dataset[int(index)] for index in indices]
            results["retained_eval"] = _from_records(
                repository=repository,
                split=split,
                revision=revision,
                image_field=str(spec["image_field"]),
                label_field=label_field,
                records=records,
                class_names=class_names,
                selection={
                    "mode": "materialized-seeded-split",
                    "seed": seed + 1000,
                    "indices_sha256": hashlib.sha256(
                        json.dumps(indices).encode("utf-8")
                    ).hexdigest(),
                    "size": len(indices),
                },
            )
    return results


def _dataset_server_rows(
    repository: str,
    split: str,
    *,
    config_name: str,
    offset: int,
    length: int,
) -> dict[str, Any]:
    import requests

    params = {
        "dataset": repository,
        "config": config_name,
        "split": split,
        "offset": offset,
        "length": length,
    }
    identity = hashlib.sha256(
        json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_path = Path("artifacts/cache/huggingface/rows") / f"{identity}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    for attempt in range(5):
        response = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params=params,
            timeout=60,
        )
        if response.status_code != 429:
            response.raise_for_status()
            payload = response.json()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        if attempt < 4:
            time.sleep(2**attempt)
    response.raise_for_status()
    raise AssertionError("unreachable")


def _server_class_names(
    payload: dict[str, Any], label_field: str = "label"
) -> tuple[str, ...]:
    for feature in payload["features"]:
        if feature["name"] == label_field:
            return tuple(feature["type"]["names"])
    raise ValueError(f"dataset server response has no {label_field} feature")


def _decode_server_records(
    rows: list[dict[str, Any]], *, image_field: str, label_field: str, revision: str
) -> list[dict[str, Any]]:
    import requests
    from PIL import Image

    decoded = []
    for item in rows:
        source = item["row"][image_field]["src"]
        if revision not in source:
            raise ValueError("dataset server asset revision does not match resolved revision")
        identity = hashlib.sha256(source.encode()).hexdigest()
        cache_path = Path("artifacts/cache/huggingface/images") / f"{identity}.bin"
        if cache_path.exists():
            content = cache_path.read_bytes()
        else:
            for attempt in range(5):
                response = requests.get(source, timeout=60)
                if response.status_code != 429:
                    response.raise_for_status()
                    content = response.content
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(content)
                    break
                if attempt < 4:
                    time.sleep(2**attempt)
            else:
                response.raise_for_status()
        image = Image.open(BytesIO(content)).convert("RGB")
        decoded.append(
            {
                image_field: image,
                label_field: int(item["row"][label_field]),
                "row_idx": int(item["row_idx"]),
            }
        )
    return decoded


def _server_dataset(
    *,
    spec: dict[str, Any],
    revision: str,
    rows: list[dict[str, Any]],
    class_names: tuple[str, ...],
    selection: dict[str, Any],
) -> ImageDataset:
    image_field = str(spec["image_field"])
    label_field = str(spec.get("label_field", "label"))
    records = _decode_server_records(
        rows, image_field=image_field, label_field=label_field, revision=revision
    )
    selection = dict(selection)
    selection["row_indices"] = [record["row_idx"] for record in records]
    return _from_records(
        repository=str(spec["repository"]),
        split=str(spec["split"]),
        revision=revision,
        image_field=image_field,
        label_field=label_field,
        records=records,
        class_names=class_names,
        selection=selection,
    )


def _ranges_overlap(left_start: int, left_size: int, right_start: int, right_size: int) -> bool:
    """Return whether two half-open row ranges share at least one example."""
    if min(left_size, right_size) < 0:
        raise ValueError("dataset range sizes must be non-negative")
    return max(left_start, right_start) < min(
        left_start + left_size, right_start + right_size
    )


def load_dataset_server_stratified(
    *, adaptation_spec: dict[str, Any], retained_spec: dict[str, Any], seed: int
) -> dict[str, ImageDataset]:
    """Fetch a small class-diverse, immutable subset without full dataset downloads."""
    import numpy as np

    adaptation_repository = str(adaptation_spec["repository"])
    adaptation_split = str(adaptation_spec["split"])
    adaptation_config = str(adaptation_spec.get("config", "default"))
    adaptation_revision = resolve_dataset_revision(adaptation_repository)
    first = _dataset_server_rows(
        adaptation_repository,
        adaptation_split,
        config_name=adaptation_config,
        offset=0,
        length=1,
    )
    total_rows = int(first["num_rows_total"])
    adaptation_label_field = str(adaptation_spec.get("label_field", "label"))
    class_names = _server_class_names(first, adaptation_label_field)
    class_count = int(adaptation_spec["class_count"])
    train_per_class = int(adaptation_spec["train_per_class"])
    eval_per_class = int(adaptation_spec["eval_per_class"])
    needed = train_per_class + eval_per_class
    window = max(int(adaptation_spec.get("server_window", 64)), needed)
    rng = np.random.default_rng(seed)
    selected: dict[int, list[dict[str, Any]]] = {}
    attempts = 0
    while len(selected) < class_count and attempts < class_count * 20:
        candidate = int(rng.integers(0, total_rows))
        offset = max(0, min(candidate - window // 2, total_rows - window))
        payload = _dataset_server_rows(
            adaptation_repository,
            adaptation_split,
            config_name=adaptation_config,
            offset=offset,
            length=window,
        )
        center = min(payload["rows"], key=lambda item: abs(item["row_idx"] - candidate))
        label = int(center["row"][adaptation_label_field])
        matching = [
            item
            for item in payload["rows"]
            if int(item["row"][adaptation_label_field]) == label
        ]
        if label not in selected and len(matching) >= needed:
            matching.sort(key=lambda item: abs(item["row_idx"] - candidate))
            selected[label] = sorted(matching[:needed], key=lambda item: item["row_idx"])
        attempts += 1
    if len(selected) != class_count:
        raise RuntimeError(
            f"could not construct {class_count} class-diverse Food-101 groups"
        )
    train_rows = []
    eval_rows = []
    for label in sorted(selected):
        train_rows.extend(selected[label][:train_per_class])
        eval_rows.extend(selected[label][train_per_class:])
    common_selection = {
        "mode": "dataset-server-stratified",
        "seed": seed,
        "class_ids": sorted(selected),
        "server_window": window,
    }
    adaptation_train = _server_dataset(
        spec=adaptation_spec,
        revision=adaptation_revision,
        rows=train_rows,
        class_names=class_names,
        selection={**common_selection, "role": "train"},
    )
    adaptation_eval = _server_dataset(
        spec=adaptation_spec,
        revision=adaptation_revision,
        rows=eval_rows,
        class_names=class_names,
        selection={**common_selection, "role": "eval"},
    )
    retained_repository = str(retained_spec["repository"])
    retained_split = str(retained_spec["split"])
    retained_config = str(retained_spec.get("config", "default"))
    retained_revision = resolve_dataset_revision(retained_repository)
    if str(retained_spec.get("selection_mode", "range")) == "stratified":
        first_retained = _dataset_server_rows(
            retained_repository,
            retained_split,
            config_name=retained_config,
            offset=0,
            length=1,
        )
        retained_total = int(first_retained["num_rows_total"])
        retained_label_field = str(retained_spec.get("label_field", "label"))
        retained_class_count = int(retained_spec["class_count"])
        retained_per_class = int(retained_spec["eval_per_class"])
        retained_window = max(
            int(retained_spec.get("server_window", 64)), retained_per_class
        )
        retained_rng = np.random.default_rng(seed + 1000)
        retained_selected: dict[int, list[dict[str, Any]]] = {}
        retained_attempts = 0
        while (
            len(retained_selected) < retained_class_count
            and retained_attempts < retained_class_count * 30
        ):
            candidate = int(retained_rng.integers(0, retained_total))
            offset = max(
                0,
                min(
                    candidate - retained_window // 2,
                    retained_total - retained_window,
                ),
            )
            payload = _dataset_server_rows(
                retained_repository,
                retained_split,
                config_name=retained_config,
                offset=offset,
                length=retained_window,
            )
            center = min(
                payload["rows"],
                key=lambda item: abs(item["row_idx"] - candidate),
            )
            label = int(center["row"][retained_label_field])
            matching = [
                item
                for item in payload["rows"]
                if int(item["row"][retained_label_field]) == label
            ]
            if label not in retained_selected and len(matching) >= retained_per_class:
                matching.sort(key=lambda item: abs(item["row_idx"] - candidate))
                retained_selected[label] = sorted(
                    matching[:retained_per_class], key=lambda item: item["row_idx"]
                )
            retained_attempts += 1
        if len(retained_selected) != retained_class_count:
            raise RuntimeError(
                f"could not construct {retained_class_count} retained class groups"
            )
        retained_rows = [
            item
            for label in sorted(retained_selected)
            for item in retained_selected[label]
        ]
        retained_eval = _server_dataset(
            spec=retained_spec,
            revision=retained_revision,
            rows=retained_rows,
            class_names=_server_class_names(
                first_retained, retained_label_field
            ),
            selection={
                "mode": "dataset-server-stratified",
                "seed": seed + 1000,
                "class_ids": sorted(retained_selected),
                "per_class": retained_per_class,
                "server_window": retained_window,
            },
        )
        return {
            "adaptation_train": adaptation_train,
            "adaptation_eval": adaptation_eval,
            "retained_eval": retained_eval,
        }
    retained_payload = _dataset_server_rows(
        retained_repository,
        retained_split,
        config_name=retained_config,
        offset=int(retained_spec["eval_start"]),
        length=int(retained_spec["eval_size"]),
    )
    retained_eval = _server_dataset(
        spec=retained_spec,
        revision=retained_revision,
        rows=retained_payload["rows"],
        class_names=_server_class_names(
            retained_payload, str(retained_spec.get("label_field", "label"))
        ),
        selection={
            "mode": "dataset-server-range",
            "start": int(retained_spec["eval_start"]),
            "size": int(retained_spec["eval_size"]),
        },
    )
    datasets = {
        "adaptation_train": adaptation_train,
        "adaptation_eval": adaptation_eval,
        "retained_eval": retained_eval,
    }
    reference_size = int(retained_spec.get("reference_size", 0))
    if reference_size:
        eval_start = int(retained_spec["eval_start"])
        eval_size = int(retained_spec["eval_size"])
        reference_start = int(retained_spec["reference_start"])
        if _ranges_overlap(eval_start, eval_size, reference_start, reference_size):
            raise ValueError("retained reference and evaluation ranges must be disjoint")
        reference_payload = _dataset_server_rows(
            retained_repository,
            retained_split,
            config_name=retained_config,
            offset=reference_start,
            length=reference_size,
        )
        datasets["retained_reference"] = _server_dataset(
            spec=retained_spec,
            revision=retained_revision,
            rows=reference_payload["rows"],
            class_names=_server_class_names(
                reference_payload, str(retained_spec.get("label_field", "label"))
            ),
            selection={
                "mode": "dataset-server-range",
                "role": "optimization-reference",
                "start": reference_start,
                "size": reference_size,
            },
        )
    return datasets


def load_reproduction_datasets(config_datasets: dict[str, Any], seed: int) -> dict[str, ImageDataset]:
    mode = config_datasets["loading_mode"]
    adaptation = dict(config_datasets["adaptation"])
    retained = dict(config_datasets["retained"])
    if mode == "streaming-prefix":
        adaptation_revision = resolve_dataset_revision(str(adaptation["repository"]))
        retained_revision = resolve_dataset_revision(str(retained["repository"]))
        datasets = {
            "adaptation_train": load_streaming_range(
                adaptation,
                start=int(adaptation["train_start"]),
                size=int(adaptation["train_size"]),
                revision=adaptation_revision,
            ),
            "adaptation_eval": load_streaming_range(
                adaptation,
                start=int(adaptation["eval_start"]),
                size=int(adaptation["eval_size"]),
                revision=adaptation_revision,
            ),
            "retained_eval": load_streaming_range(
                retained,
                start=int(retained["eval_start"]),
                size=int(retained["eval_size"]),
                revision=retained_revision,
            ),
        }
        reference_size = int(retained.get("reference_size", 0))
        if reference_size:
            eval_start = int(retained["eval_start"])
            eval_size = int(retained["eval_size"])
            reference_start = int(retained["reference_start"])
            if _ranges_overlap(eval_start, eval_size, reference_start, reference_size):
                raise ValueError("retained reference and evaluation ranges must be disjoint")
            datasets["retained_reference"] = load_streaming_range(
                retained,
                start=reference_start,
                size=reference_size,
                revision=retained_revision,
            )
        return datasets
    if mode == "materialized-seeded-split":
        return load_materialized_seeded_splits(
            adaptation_spec=adaptation, retained_spec=retained, seed=seed
        )
    if mode == "dataset-server-stratified":
        return load_dataset_server_stratified(
            adaptation_spec=adaptation, retained_spec=retained, seed=seed
        )
    raise ValueError(f"unsupported dataset loading_mode: {mode}")
