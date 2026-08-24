"""Create rho-n CutMix and keyed block-permuted training samples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

from privacy_pipeline.config import load_config
from privacy_pipeline.transforms import KeyedBlockPermutation, RhoNCutMix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if not config.defense.enabled:
        raise ValueError("Raw configurations do not require preprocessing.")
    set_seed(config.seed)

    dataset = datasets.ImageFolder(
        config.dataset.root,
        transform=transforms.Compose(
            [
                transforms.Resize(
                    (config.dataset.image_size, config.dataset.image_size)
                ),
                transforms.ToTensor(),
            ]
        ),
    )
    mixer = RhoNCutMix(
        dataset=dataset,
        targets=dataset.targets,
        num_classes=config.dataset.num_classes,
        rho=config.defense.rho,
        donors=config.defense.donors,
        minimum_patch_size=config.defense.minimum_patch_size,
        seed=config.seed,
    )
    permutation = KeyedBlockPermutation(
        grid_size=config.defense.grid_size,
        codebook_size=config.defense.codebook_size,
        root_key=config.defense.root_key,
    )

    output_root = config.dataset.prepared_root
    image_root = output_root / "images"
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_root} already exists. Use --overwrite to replace matching files."
        )
    image_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for index in range(len(dataset)):
        mixed = mixer(index)
        scrambled, permutation_index = permutation.apply(mixed.image, index)
        relative_path = Path("images") / f"{index:08d}.jpg"
        to_pil_image(scrambled.clamp(0, 1)).save(
            output_root / relative_path,
            quality=95,
        )
        rows.append(
            {
                "image": relative_path.as_posix(),
                "soft_label": json.dumps(mixed.soft_label.tolist()),
                "permutation_index": permutation_index,
                "donor_indices": json.dumps(mixed.donor_indices),
            }
        )

    pd.DataFrame(rows).to_csv(output_root / "manifest.csv", index=False)
    print(f"Prepared {len(rows)} samples in {output_root}")


if __name__ == "__main__":
    main()
