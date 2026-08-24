"""Shared command-line helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from privacy_pipeline.config import ExperimentConfig
from privacy_pipeline.data import PreparedPrivacyDataset, RawImageFolder
from privacy_pipeline.models import build_model
from privacy_pipeline.transforms import KeyedBlockPermutation


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def dataset_for(config: ExperimentConfig, root: str | Path):
    cls = PreparedPrivacyDataset if config.defense.enabled else RawImageFolder
    return cls(root, config.dataset.image_size)


def load_model(config: ExperimentConfig, checkpoint: str | Path, device: torch.device):
    codebook = None
    if config.defense.enabled:
        codebook = KeyedBlockPermutation(
            config.defense.grid_size,
            config.defense.codebook_size,
            config.defense.root_key,
        ).codebook
    model = build_model(
        config.model.backbone,
        config.dataset.num_classes,
        pretrained=False,
        codebook=codebook,
        grid_size=config.defense.grid_size,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


def loader_for(config: ExperimentConfig, root: str | Path, shuffle: bool = False):
    return DataLoader(
        dataset_for(config, root),
        batch_size=config.federated.batch_size,
        shuffle=shuffle,
    )
