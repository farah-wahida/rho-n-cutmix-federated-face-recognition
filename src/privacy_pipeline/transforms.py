from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MixedSample:
    image: torch.Tensor
    soft_label: torch.Tensor
    donor_indices: tuple[int, ...]


def _as_batch(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if images.ndim == 3:
        return images.unsqueeze(0), True
    if images.ndim != 4:
        raise ValueError("Expected an image tensor with shape [C,H,W] or [B,C,H,W].")
    return images, False


def permute_blocks(
    images: torch.Tensor,
    permutation: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    batch, squeezed = _as_batch(images)
    batch_size, channels, height, width = batch.shape
    if height % grid_size or width % grid_size:
        raise ValueError("Image dimensions must be divisible by grid_size.")

    block_height = height // grid_size
    block_width = width // grid_size
    block_count = grid_size * grid_size
    blocks = (
        batch.reshape(
            batch_size,
            channels,
            grid_size,
            block_height,
            grid_size,
            block_width,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch_size, block_count, channels, block_height, block_width)
    )

    permutation = permutation.to(device=batch.device, dtype=torch.long)
    if permutation.ndim == 1:
        selected = blocks[:, permutation]
    elif permutation.ndim == 2 and permutation.shape[0] == batch_size:
        gather_index = permutation[:, :, None, None, None].expand_as(blocks)
        selected = torch.gather(blocks, dim=1, index=gather_index)
    else:
        raise ValueError("Permutation must have shape [L] or [B,L].")

    output = (
        selected.reshape(
            batch_size,
            grid_size,
            grid_size,
            channels,
            block_height,
            block_width,
        )
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch_size, channels, height, width)
    )
    return output.squeeze(0) if squeezed else output


def inverse_block_permutation(
    images: torch.Tensor,
    permutation: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    return permute_blocks(images, torch.argsort(permutation, dim=-1), grid_size)


class KeyedBlockPermutation:
    def __init__(self, grid_size: int, codebook_size: int, root_key: str):
        if grid_size < 1 or codebook_size < 1:
            raise ValueError("grid_size and codebook_size must be positive.")
        self.grid_size = grid_size
        self.codebook_size = codebook_size
        self.root_key = root_key
        self.codebook = self._build_codebook()

    def _seed(self, value: str) -> int:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def _build_codebook(self) -> torch.Tensor:
        block_count = self.grid_size**2
        permutations = []
        seen = set()
        counter = 0
        while len(permutations) < self.codebook_size:
            generator = torch.Generator().manual_seed(
                self._seed(f"{self.root_key}:codebook:{counter}")
            )
            permutation = torch.randperm(block_count, generator=generator)
            signature = tuple(permutation.tolist())
            if signature not in seen:
                permutations.append(permutation)
                seen.add(signature)
            counter += 1
        return torch.stack(permutations)

    def index_for_sample(self, sample_index: int) -> int:
        return self._seed(f"{self.root_key}:sample:{sample_index}") % self.codebook_size

    def apply(self, image: torch.Tensor, sample_index: int) -> tuple[torch.Tensor, int]:
        index = self.index_for_sample(sample_index)
        return permute_blocks(image, self.codebook[index], self.grid_size), index

    def invert(self, image: torch.Tensor, permutation_index: int) -> torch.Tensor:
        return inverse_block_permutation(
            image,
            self.codebook[permutation_index],
            self.grid_size,
        )


class RhoNCutMix:
    def __init__(
        self,
        dataset,
        targets: Sequence[int],
        num_classes: int,
        rho: float,
        donors: int,
        minimum_patch_size: int = 1,
        seed: int = 42,
    ):
        if not 0 < rho <= 1:
            raise ValueError("rho must be in the interval (0, 1].")
        if donors < 1:
            raise ValueError("donors must be positive.")
        self.dataset = dataset
        self.targets = [int(target) for target in targets]
        self.num_classes = num_classes
        self.rho = rho
        self.donors = donors
        self.minimum_patch_size = minimum_patch_size
        self.seed = seed
        self.class_to_indices: dict[int, list[int]] = {}
        for index, target in enumerate(self.targets):
            self.class_to_indices.setdefault(target, []).append(index)

    def _donor_indices(self, base_class: int, rng: random.Random) -> list[int]:
        available = [label for label in self.class_to_indices if label != base_class]
        if len(available) < self.donors:
            raise ValueError("The dataset does not contain enough distinct donor classes.")
        donor_classes = rng.sample(available, self.donors)
        return [rng.choice(self.class_to_indices[label]) for label in donor_classes]

    def __call__(self, sample_index: int) -> MixedSample:
        rng = random.Random(self.seed + sample_index)
        np_rng = np.random.default_rng(self.seed + sample_index)
        base_image, base_class = self.dataset[sample_index]
        donor_indices = self._donor_indices(int(base_class), rng)
        proportions = np_rng.dirichlet(np.ones(self.donors))

        composite = base_image.clone()
        soft_label = torch.zeros(self.num_classes, dtype=torch.float32)
        _, height, width = composite.shape

        for proportion, donor_index in zip(proportions, donor_indices):
            donor_image, donor_class = self.dataset[donor_index]
            requested_area = float(proportion) * self.rho * height * width
            side = int(round(math.sqrt(max(requested_area, 1.0))))
            side = max(self.minimum_patch_size, min(side, height, width))
            top = rng.randint(0, height - side)
            left = rng.randint(0, width - side)
            composite[:, top : top + side, left : left + side] = donor_image[
                :, top : top + side, left : left + side
            ]
            soft_label[int(donor_class)] += float(proportion)

        soft_label = F.normalize(soft_label, p=1, dim=0)
        return MixedSample(composite, soft_label, tuple(donor_indices))
