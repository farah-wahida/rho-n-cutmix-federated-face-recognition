"""Validate publication-matrix and reference-result coverage."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_DATASETS = {"pubfig", "lfw", "pins", "cifar10"}
EXPECTED_BACKBONES = {"resnet18", "resnet34"}
EXPECTED_SETTINGS = {"raw", "defense_a", "defense_b", "defense_c", "defense_d"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="configs/paper_grid.yaml")
    parser.add_argument("--results", default="results/paper")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                sys.executable,
                "scripts/generate_configs.py",
                "--grid",
                args.grid,
                "--output",
                directory,
            ],
            check=True,
        )
        configs = list(Path(directory).glob("*/*.yaml"))
        if len(configs) != 40:
            raise SystemExit(f"Expected 40 generated configs, found {len(configs)}")

    result_root = Path(args.results)
    required = {
        "table_iv_accuracy.csv",
        "table_v_mia_max_softmax.csv",
        "table_vi_mia_scores.csv",
        "table_vii_inversion.csv",
        "table_viii_nll.csv",
        "table_ix_verification.csv",
        "table_x_latency.csv",
    }
    missing = sorted(name for name in required if not (result_root / name).is_file())
    if missing:
        raise SystemExit("Missing reference tables: " + ", ".join(missing))

    with (result_root / "table_iv_accuracy.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed = {
        (row["dataset"], row["backbone"], row["configuration"]) for row in rows
    }
    expected = {
        (dataset, backbone, setting)
        for dataset in EXPECTED_DATASETS
        for backbone in EXPECTED_BACKBONES
        for setting in EXPECTED_SETTINGS
    }
    if observed != expected:
        raise SystemExit("Table IV does not cover the exact 40-run matrix.")

    print("Publication release validation passed: 40 configs and seven reference tables.")


if __name__ == "__main__":
    main()
