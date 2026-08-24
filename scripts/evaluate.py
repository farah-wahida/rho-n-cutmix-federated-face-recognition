"""Evaluate recognition utility and validation-fitted calibration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from privacy_pipeline.config import load_config
from privacy_pipeline.data import PreparedPrivacyDataset, RawImageFolder
from privacy_pipeline.evaluation import (
    classification_metrics,
    collect_logits,
    fit_temperature,
)
from privacy_pipeline.models import build_model
from privacy_pipeline.transforms import KeyedBlockPermutation


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--validation-root")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.device)
    data_root = Path(args.data_root) if args.data_root else (
        config.dataset.prepared_root
        if config.defense.enabled
        else config.dataset.root
    )
    dataset_type = PreparedPrivacyDataset if config.defense.enabled else RawImageFolder
    dataset = dataset_type(data_root, config.dataset.image_size)

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
    checkpoint = Path(args.checkpoint) if args.checkpoint else config.output_dir / "model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    logits, targets = collect_logits(
        model,
        DataLoader(dataset, batch_size=config.federated.batch_size, shuffle=False),
        device,
    )
    raw = classification_metrics(logits, targets, ece_bins=config.evaluation.ece_bins)
    results: dict[str, object] = {"raw": asdict(raw)}

    if config.evaluation.temperature_scaling:
        if not args.validation_root:
            raise ValueError(
                "--validation-root is required when temperature scaling is enabled."
            )
        validation_set = dataset_type(args.validation_root, config.dataset.image_size)
        validation_logits, validation_targets = collect_logits(
            model,
            DataLoader(
                validation_set,
                batch_size=config.federated.batch_size,
                shuffle=False,
            ),
            device,
        )
        temperature = fit_temperature(validation_logits, validation_targets)
        calibrated = classification_metrics(
            logits, targets, temperature, ece_bins=config.evaluation.ece_bins
        )
        results.update(
            {
                "temperature": temperature,
                "temperature_scaled": asdict(calibrated),
            }
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "evaluation.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved evaluation metrics to {output_path}")


if __name__ == "__main__":
    main()
