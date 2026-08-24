"""Evaluate EER and TAR at the FAR values reported in the paper."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from common import load_model, loader_for, resolve_device
from privacy_pipeline.config import load_config
from privacy_pipeline.evaluation import collect_embeddings, verification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else config.output_dir / "model.pt"
    model = load_model(config, checkpoint, device)
    embeddings, labels = collect_embeddings(
        model, loader_for(config, args.data_root), device
    )
    metrics = verification_metrics(
        embeddings, labels, config.evaluation.verification_fars
    )
    output = config.output_dir / "verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    print(f"Saved verification metrics to {output}")


if __name__ == "__main__":
    main()
