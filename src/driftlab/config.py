from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the contract."""


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    tier: str
    seed: int
    model: dict[str, Any]
    method: dict[str, Any]
    datasets: dict[str, Any]
    training: dict[str, Any]
    analysis: dict[str, Any]
    runtime: dict[str, Any]
    output_dir: Path

    @property
    def total_steps(self) -> int:
        return int(self.training["total_steps"])

    @property
    def checkpoint_steps(self) -> tuple[int, ...]:
        return tuple(int(step) for step in self.training["checkpoint_steps"])

    @property
    def learning_rate(self) -> float:
        return float(self.training["learning_rate"])

    @property
    def batch_size(self) -> int:
        return int(self.training["batch_size"])

    def scientific_payload(self) -> dict[str, Any]:
        """Return fields that define scientific identity, excluding storage paths."""
        return {
            "name": self.name,
            "tier": self.tier,
            "seed": self.seed,
            "model": self.model,
            "method": self.method,
            "datasets": self.datasets,
            "training": self.training,
            "analysis": self.analysis,
            "runtime": self.runtime,
        }

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            self.scientific_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    @property
    def run_id(self) -> str:
        return f"{self.name}-{self.config_hash}-s{self.seed}"


def _validate(config: ExperimentConfig) -> None:
    if not config.name or any(char.isspace() for char in config.name):
        raise ConfigError("name must be a non-empty slug without whitespace")
    if config.tier not in {"smoke", "portfolio", "research"}:
        raise ConfigError("tier must be smoke, portfolio, or research")
    if config.seed < 0:
        raise ConfigError("seed must be non-negative")
    if config.total_steps < 1:
        raise ConfigError("training.total_steps must be positive")
    steps = config.checkpoint_steps
    if not steps or steps[0] != 0:
        raise ConfigError("training.checkpoint_steps must start at 0")
    if steps[-1] != config.total_steps:
        raise ConfigError("training.checkpoint_steps must end at total_steps")
    if tuple(sorted(set(steps))) != steps:
        raise ConfigError("training.checkpoint_steps must be sorted and unique")
    if config.batch_size < 2:
        raise ConfigError("training.batch_size must be at least 2")
    if config.learning_rate <= 0:
        raise ConfigError("training.learning_rate must be positive")
    for field, mapping in (
        ("model", config.model),
        ("method", config.method),
        ("datasets", config.datasets),
    ):
        if not mapping:
            raise ConfigError(f"{field} must not be empty")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ConfigError("configuration root must be a mapping")
    training = _require_mapping(raw.get("training"), "training")
    required_training = {"total_steps", "checkpoint_steps", "batch_size", "learning_rate"}
    missing = sorted(required_training - training.keys())
    if missing:
        raise ConfigError(f"training is missing fields: {', '.join(missing)}")
    config = ExperimentConfig(
        name=str(raw.get("name", "")),
        tier=str(raw.get("tier", "")),
        seed=int(raw.get("seed", -1)),
        model=_require_mapping(raw.get("model"), "model"),
        method=_require_mapping(raw.get("method"), "method"),
        datasets=_require_mapping(raw.get("datasets"), "datasets"),
        training=training,
        analysis=_require_mapping(raw.get("analysis", {}), "analysis"),
        runtime=_require_mapping(raw.get("runtime", {}), "runtime"),
        output_dir=Path(str(raw.get("output_dir", "artifacts/runs"))),
    )
    _validate(config)
    return config
