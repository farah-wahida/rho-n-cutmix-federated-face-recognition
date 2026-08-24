from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import yaml


def test_publication_grid_generates_40_configs(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_configs.py",
            "--output",
            str(tmp_path),
        ],
        check=True,
    )
    paths = list(tmp_path.glob("*/*.yaml"))
    assert len(paths) == 40

    pubfig_a = yaml.safe_load(
        (tmp_path / "pubfig" / "defense_a_resnet18.yaml").read_text()
    )
    assert pubfig_a["federated"]["confidence_penalty_weight"] == 0.05
    assert pubfig_a["evaluation"]["inversion_restarts"] == 3


def test_reference_accuracy_covers_complete_matrix() -> None:
    path = Path("results/paper/table_iv_accuracy.csv")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
    assert {row["dataset"] for row in rows} == {"pubfig", "lfw", "pins", "cifar10"}
    assert {row["backbone"] for row in rows} == {"resnet18", "resnet34"}
    assert {row["configuration"] for row in rows} == {
        "raw",
        "defense_a",
        "defense_b",
        "defense_c",
        "defense_d",
    }
