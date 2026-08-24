"""Train one federated experiment from a YAML configuration."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from privacy_pipeline.config import load_config
from privacy_pipeline.data import PreparedPrivacyDataset, RawImageFolder, partition_dirichlet
from privacy_pipeline.federated import federated_train
from privacy_pipeline.models import build_model
from privacy_pipeline.transforms import KeyedBlockPermutation


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def dataset_targets(dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return [int(value) for value in dataset.targets]
    return [
        int(torch.tensor(json.loads(value)).argmax())
        for value in dataset.manifest["soft_label"]
    ]


def load_dataset(root: str | Path, image_size: int, defended: bool):
    cls = PreparedPrivacyDataset if defended else RawImageFolder
    return cls(root, image_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--validation-root")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    dataset = load_dataset(
        config.dataset.prepared_root if config.defense.enabled else config.dataset.root,
        config.dataset.image_size,
        config.defense.enabled,
    )

    codebook = None
    if config.defense.enabled:
        codebook = KeyedBlockPermutation(
            config.defense.grid_size,
            config.defense.codebook_size,
            config.defense.root_key,
        ).codebook

    targets = dataset_targets(dataset)
    clients = partition_dirichlet(
        targets,
        client_count=config.federated.clients,
        alpha=config.federated.dirichlet_alpha,
        seed=config.seed,
    )
    model = build_model(
        config.model.backbone,
        config.dataset.num_classes,
        config.model.pretrained,
        codebook=codebook,
        grid_size=config.defense.grid_size,
    )
    validation_loader = None
    if args.validation_root:
        validation_set = load_dataset(
            args.validation_root,
            config.dataset.image_size,
            config.defense.enabled,
        )
        validation_loader = DataLoader(
            validation_set, batch_size=config.federated.batch_size, shuffle=False
        )

    history = federated_train(
        model,
        dataset,
        clients,
        rounds=config.federated.rounds,
        local_epochs=config.federated.local_epochs,
        batch_size=config.federated.batch_size,
        learning_rate=config.federated.learning_rate,
        minimum_learning_rate=config.federated.minimum_learning_rate,
        weight_decay=config.federated.weight_decay,
        gradient_clip_norm=config.federated.gradient_clip_norm,
        confidence_penalty_weight=config.federated.confidence_penalty_weight,
        validation_loader=validation_loader,
        early_stopping_patience=config.federated.early_stopping_patience,
        device=device,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.output_dir / "model.pt")
    (config.output_dir / "training_history.json").write_text(
        json.dumps([asdict(item) for item in history], indent=2),
        encoding="utf-8",
    )
    print(f"Saved model and training history to {config.output_dir}")


if __name__ == "__main__":
    main()
