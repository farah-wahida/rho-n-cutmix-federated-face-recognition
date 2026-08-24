"""Evaluate recognition utility and calibration."""

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
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.device)
    data_root = Path(args.data_root) if args.data_root else (
        config.dataset.prepared_root
        if config.defense.enabled
        else config.dataset.root
    )

    if config.defense.enabled:
        dataset = PreparedPrivacyDataset(data_root, config.dataset.image_size)
        keyed_permutation = KeyedBlockPermutation(
            config.defense.grid_size,
            config.defense.codebook_size,
            config.defense.root_key,
        )
        codebook = keyed_permutation.codebook
    else:
        dataset = RawImageFolder(data_root, config.dataset.image_size)
        codebook = None

    model = build_model(
        config.model.backbone,
        config.dataset.num_classes,
        pretrained=False,
        codebook=codebook,
        grid_size=config.defense.grid_size,
    )
    checkpoint = Path(args.checkpoint) if args.checkpoint else config.output_dir / "model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    loader = DataLoader(
        dataset,
        batch_size=config.federated.batch_size,
        shuffle=False,
    )
    logits, targets = collect_logits(model, loader, device)
    raw = classification_metrics(logits, targets)

    results: dict[str, object] = {"raw": asdict(raw)}
    if config.evaluation.temperature_scaling:
        temperature = fit_temperature(logits, targets)
        calibrated = classification_metrics(logits, targets, temperature)
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
