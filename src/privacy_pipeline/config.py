from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    prepared_root: Path
    image_size: int
    num_classes: int


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    pretrained: bool


@dataclass(frozen=True)
class DefenseConfig:
    enabled: bool
    rho: float
    donors: int
    minimum_patch_size: int
    grid_size: int
    codebook_size: int
    root_key: str


@dataclass(frozen=True)
class FederatedConfig:
    clients: int
    rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    minimum_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    early_stopping_patience: int
    dirichlet_alpha: float
    confidence_penalty_weight: float


@dataclass(frozen=True)
class EvaluationConfig:
    validation_fraction: float
    temperature_scaling: bool
    ece_bins: int
    verification_fars: tuple[float, ...]
    inversion_targets: int
    inversion_restarts: int
    inversion_iterations: int
    inversion_learning_rate: float
    inversion_tv_weight: float
    inversion_l2_weight: float
    inversion_permutation_entropy_weight: float
    inversion_sinkhorn_iterations: int


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    device: str
    dataset: DatasetConfig
    model: ModelConfig
    defense: DefenseConfig
    federated: FederatedConfig
    evaluation: EvaluationConfig
    output_dir: Path


def _build(section: type, values: dict[str, Any]):
    converted = dict(values)
    for key in ("root", "prepared_root", "output_dir"):
        if key in converted:
            converted[key] = Path(converted[key])
    if section is EvaluationConfig:
        converted["verification_fars"] = tuple(converted["verification_fars"])
    return section(**converted)


def load_config(path: str | Path) -> ExperimentConfig:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        seed=int(values["seed"]),
        device=str(values.get("device", "auto")),
        dataset=_build(DatasetConfig, values["dataset"]),
        model=_build(ModelConfig, values["model"]),
        defense=_build(DefenseConfig, values["defense"]),
        federated=_build(FederatedConfig, values["federated"]),
        evaluation=_build(EvaluationConfig, values["evaluation"]),
        output_dir=Path(values["output_dir"]),
    )
