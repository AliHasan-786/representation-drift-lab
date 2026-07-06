from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArrayDataset:
    name: str
    split: str
    features: np.ndarray
    labels: np.ndarray
    class_names: tuple[str, ...]
    fingerprint: str


def _fingerprint(
    name: str,
    split: str,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    metadata = json.dumps(
        {"name": name, "split": split, "class_names": class_names},
        sort_keys=True,
    ).encode("utf-8")
    digest.update(metadata)
    digest.update(np.ascontiguousarray(features).tobytes())
    digest.update(np.ascontiguousarray(labels).tobytes())
    return digest.hexdigest()


def make_synthetic_dataset(
    *,
    name: str,
    split: str,
    seed: int,
    samples_per_class: int,
    rotation_degrees: float = 0.0,
    noise: float = 0.18,
) -> ArrayDataset:
    """Create a deterministic three-class feature dataset without downloads."""
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2")
    rng = np.random.default_rng(seed)
    angles = np.deg2rad(np.array([15.0, 135.0, 255.0]))
    centers = np.column_stack([np.cos(angles), np.sin(angles)])
    theta = np.deg2rad(rotation_degrees)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    centers = centers @ rotation.T
    features = []
    labels = []
    for label, center in enumerate(centers):
        features.append(center + rng.normal(0.0, noise, (samples_per_class, 2)))
        labels.extend([label] * samples_per_class)
    feature_array = np.vstack(features).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(label_array))
    feature_array = feature_array[order]
    label_array = label_array[order]
    class_names = ("amber", "violet", "cyan")
    return ArrayDataset(
        name=name,
        split=split,
        features=feature_array,
        labels=label_array,
        class_names=class_names,
        fingerprint=_fingerprint(name, split, feature_array, label_array, class_names),
    )


def make_smoke_splits(
    seed: int, *, adaptation_rotation_degrees: float = 85.0
) -> dict[str, ArrayDataset]:
    """Return isolated train/evaluation splits for the CPU smoke experiment."""
    return {
        "adaptation_train": make_synthetic_dataset(
            name="synthetic-rotated-v1",
            split="train",
            seed=seed + 11,
            samples_per_class=48,
            rotation_degrees=adaptation_rotation_degrees,
        ),
        "adaptation_eval": make_synthetic_dataset(
            name="synthetic-rotated-v1",
            split="eval",
            seed=seed + 12,
            samples_per_class=30,
            rotation_degrees=adaptation_rotation_degrees,
        ),
        "retained_eval": make_synthetic_dataset(
            name="synthetic-reference-v1",
            split="eval",
            seed=seed + 21,
            samples_per_class=30,
            rotation_degrees=0.0,
        ),
    }
