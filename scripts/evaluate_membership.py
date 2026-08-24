"""Evaluate score-based membership inference on member and non-member sets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from privacy_pipeline.attacks import membership_scores
from privacy_pipeline.config import load_config
from privacy_pipeline.data import PreparedPrivacyDataset, RawImageFolder
from privacy_pipeline.evaluation import collect_logits, membership_metrics
from privacy_pipeline.models import build_model
from privacy_pipeline.transforms import KeyedBlockPermutation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--member-root", required=True)
    parser.add_argument("--nonmember-root", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_type = PreparedPrivacyDataset if config.defense.enabled else RawImageFolder
    member_set = dataset_type(args.member_root, config.dataset.image_size)
    nonmember_set = dataset_type(args.nonmember_root, config.dataset.image_size)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    member_logits, member_targets = collect_logits(
        model,
        DataLoader(member_set, batch_size=config.federated.batch_size),
        device,
    )
    nonmember_logits, nonmember_targets = collect_logits(
        model,
        DataLoader(nonmember_set, batch_size=config.federated.batch_size),
        device,
    )
    member = membership_scores(member_logits, member_targets)
    nonmember = membership_scores(nonmember_logits, nonmember_targets)

    results = {
        name: asdict(
            membership_metrics(
                member[name].tolist(),
                nonmember[name].tolist(),
            )
        )
        for name in member
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "membership.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved membership metrics to {output_path}")


if __name__ == "__main__":
    main()
