from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class PreparedPrivacyDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int,
        normalize: bool = True,
    ):
        self.root = Path(root)
        self.manifest = pd.read_csv(self.root / "manifest.csv")
        transform_steps = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        if normalize:
            transform_steps.append(
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                )
            )
        self.transform = transforms.Compose(transform_steps)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        image = Image.open(self.root / row["image"]).convert("RGB")
        soft_label = torch.tensor(json.loads(row["soft_label"]), dtype=torch.float32)
        permutation_index = int(row["permutation_index"])
        return self.transform(image), soft_label, permutation_index


class RawImageFolder(Dataset):
    def __init__(self, root: str | Path, image_size: int):
        self.dataset = datasets.ImageFolder(
            root,
            transform=transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            ),
        )
        self.targets = self.dataset.targets
        self.classes = self.dataset.classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        soft_label = torch.nn.functional.one_hot(
            torch.tensor(target),
            num_classes=len(self.classes),
        ).float()
        return image, soft_label, -1


def partition_dirichlet(
    targets: list[int],
    client_count: int,
    alpha: float,
    seed: int,
) -> list[list[int]]:
    if client_count < 1 or alpha <= 0:
        raise ValueError("client_count and alpha must be positive.")

    rng = np.random.default_rng(seed)
    targets_array = np.asarray(targets)
    clients: list[list[int]] = [[] for _ in range(client_count)]

    for class_id in np.unique(targets_array):
        class_indices = np.flatnonzero(targets_array == class_id)
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(client_count, alpha))
        boundaries = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        partitions = np.split(class_indices, boundaries)
        for client_indices, partition in zip(clients, partitions):
            client_indices.extend(partition.tolist())

    for client_indices in clients:
        rng.shuffle(client_indices)
    return clients
