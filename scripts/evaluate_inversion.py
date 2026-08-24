"""Run the adaptive no-key inversion protocol reported in the paper."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch
from torchvision.utils import save_image

from common import load_model, resolve_device
from privacy_pipeline.attacks import adaptive_full_access_inversion
from privacy_pipeline.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--targets", nargs="*", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else config.output_dir / "model.pt"
    model = load_model(config, checkpoint, device)
    targets = args.targets or list(
        range(min(config.evaluation.inversion_targets, config.dataset.num_classes))
    )
    image_shape = (3, config.dataset.image_size, config.dataset.image_size)
    output_dir = config.output_dir / "inversion"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # The attacker knows the trained classifier and defense design but not the
    # master key or sample-specific permutation index.
    logits_fn = lambda images: model.backbone(images)
    for target in targets:
        result = adaptive_full_access_inversion(
            logits_fn,
            target_class=target,
            image_shape=image_shape,
            grid_size=config.defense.grid_size,
            iterations=config.evaluation.inversion_iterations,
            restarts=config.evaluation.inversion_restarts,
            learning_rate=config.evaluation.inversion_learning_rate,
            tv_weight=config.evaluation.inversion_tv_weight,
            l2_weight=config.evaluation.inversion_l2_weight,
            entropy_weight=config.evaluation.inversion_permutation_entropy_weight,
            sinkhorn_iterations=config.evaluation.inversion_sinkhorn_iterations,
            device=device,
        )
        save_image(result.image, output_dir / f"target_{target:03d}.png")
        row = asdict(result)
        row.pop("image")
        rows.append({"target_class": target, **row})

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} inversion results to {output_dir}")


if __name__ == "__main__":
    main()
