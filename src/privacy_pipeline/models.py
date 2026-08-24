from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    resnet18,
    resnet34,
)

from .transforms import inverse_block_permutation


class InversePermutationDecoder(nn.Module):
    def __init__(self, codebook: torch.Tensor, grid_size: int):
        super().__init__()
        self.register_buffer("codebook", codebook.long())
        self.grid_size = grid_size

    def forward(
        self,
        images: torch.Tensor,
        permutation_indices: torch.Tensor,
    ) -> torch.Tensor:
        decoded = torch.empty_like(images)
        for index in torch.unique(permutation_indices).tolist():
            mask = permutation_indices == index
            decoded[mask] = inverse_block_permutation(
                images[mask],
                self.codebook[int(index)],
                self.grid_size,
            )
        return decoded


class AuthorizedClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, decoder: InversePermutationDecoder | None):
        super().__init__()
        self.decoder = decoder
        self.backbone = backbone

    def forward(
        self,
        images: torch.Tensor,
        permutation_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.decoder is not None:
            if permutation_indices is None:
                raise ValueError("Defended models require permutation indices.")
            images = self.decoder(images, permutation_indices)
        return self.backbone(images)


def build_backbone(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    normalized_name = name.lower().replace("-", "")
    if normalized_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    elif normalized_name == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(
    backbone_name: str,
    num_classes: int,
    pretrained: bool,
    codebook: torch.Tensor | None = None,
    grid_size: int = 8,
) -> AuthorizedClassifier:
    backbone = build_backbone(backbone_name, num_classes, pretrained)
    decoder = (
        InversePermutationDecoder(codebook=codebook, grid_size=grid_size)
        if codebook is not None
        else None
    )
    return AuthorizedClassifier(backbone=backbone, decoder=decoder)
