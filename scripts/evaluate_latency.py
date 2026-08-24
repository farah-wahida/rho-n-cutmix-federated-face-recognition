"""Measure preprocessing and inference latency in milliseconds per image."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from common import load_model, loader_for, resolve_device
from privacy_pipeline.config import load_config


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_average(fn, repetitions: int, device: torch.device) -> float:
    for _ in range(10):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repetitions):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else config.output_dir / "model.pt"
    model = load_model(config, checkpoint, device)
    images, _, indices = next(iter(loader_for(config, args.data_root)))
    images, indices = images.to(device), indices.to(device)
    selected_indices = None if torch.all(indices < 0) else indices

    with torch.no_grad():
        decode_ms = timed_average(
            lambda: model.decode(images, selected_indices),
            args.repetitions,
            device,
        ) / images.shape[0]
        inference_ms = timed_average(
            lambda: model(images, selected_indices),
            args.repetitions,
            device,
        ) / images.shape[0]

    output = config.output_dir / "latency.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "device": str(device),
                "batch_size": int(images.shape[0]),
                "repetitions": args.repetitions,
                "preprocessing_ms_per_image": decode_ms,
                "inference_ms_per_image": inference_ms,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved latency metrics to {output}")


if __name__ == "__main__":
    main()
