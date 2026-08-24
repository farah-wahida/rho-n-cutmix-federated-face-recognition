"""Generate the 40 concrete configurations used by the publication."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def nested_get(values: dict, *keys: str, default=0.0):
    current = values
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="configs/paper_grid.yaml")
    parser.add_argument("--output", default="configs/generated")
    args = parser.parse_args()

    grid = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8"))
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    generated = 0
    for dataset_name, dataset in grid["datasets"].items():
        for backbone in grid["backbones"]:
            for defense_name, defense in grid["defenses"].items():
                prepared_root = (
                    f"data/prepared/{dataset_name}/train/{defense_name}"
                    if defense["enabled"]
                    else dataset["root"]
                )
                federated = dict(grid["federated"])
                federated["confidence_penalty_weight"] = float(
                    nested_get(
                        grid.get("confidence_penalty_overrides", {}),
                        dataset_name,
                        backbone,
                        defense_name,
                        default=federated["confidence_penalty_weight"],
                    )
                )
                values = {
                    "seed": grid["seed"],
                    "device": grid["device"],
                    "dataset": {
                        "name": dataset_name,
                        "root": dataset["root"],
                        "prepared_root": prepared_root,
                        "image_size": dataset["image_size"],
                        "num_classes": dataset["num_classes"],
                    },
                    "model": {"backbone": backbone, "pretrained": True},
                    "defense": {
                        **defense,
                        "minimum_patch_size": dataset["minimum_patch_size"],
                        "grid_size": grid["permutation"]["grid_size"],
                        "codebook_size": grid["permutation"]["codebook_size"],
                        "root_key": f"{dataset_name}_{grid['permutation']['root_key_suffix']}",
                    },
                    "federated": federated,
                    "evaluation": grid["evaluation"],
                    "output_dir": (
                        f"outputs/{dataset_name}/{defense_name}/{backbone}/seed{grid['seed']}"
                    ),
                }
                path = output_root / dataset_name / f"{defense_name}_{backbone}.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
                generated += 1

    print(f"Generated {generated} publication configurations under {output_root}")


if __name__ == "__main__":
    main()
