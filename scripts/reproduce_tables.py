"""Aggregate experiment JSON files into a compact CSV table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten(prefix: str, value, output: dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, child, output)
    else:
        output[prefix] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--name", default="evaluation.json")
    parser.add_argument("--output", default="results/summary.csv")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    root = Path(args.outputs)
    for path in sorted(root.rglob(args.name)):
        row: dict[str, object] = {"run": path.parent.relative_to(root).as_posix()}
        flatten("", json.loads(path.read_text(encoding="utf-8")), row)
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Aggregated {len(rows)} runs into {output_path}")


if __name__ == "__main__":
    main()
