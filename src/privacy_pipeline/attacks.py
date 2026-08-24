"""Attack utilities for privacy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass
class InversionResult:
    image: Tensor
    target_probability: float
    target_success: bool
    objective: float
    restart_index: int
    iterations_run: int


def membership_scores(logits: Tensor, soft_targets: Tensor) -> dict[str, Tensor]:
    labels = soft_targets.argmax(dim=1)
    probabilities = torch.softmax(logits, dim=1)
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    return {
        "max_softmax": probabilities.max(dim=1).values,
        "negative_loss": -losses,
        "negative_entropy": -entropy,
    }


def sinkhorn(logits: Tensor, iterations: int = 10, temperature: float = 1.0) -> Tensor:
    matrix = logits / max(temperature, 1e-6)
    for _ in range(iterations):
        matrix = matrix - torch.logsumexp(matrix, dim=-1, keepdim=True)
        matrix = matrix - torch.logsumexp(matrix, dim=-2, keepdim=True)
    return matrix.exp()


def soft_permute_blocks(images: Tensor, matrix: Tensor, grid_size: int) -> Tensor:
    batch_size, channels, height, width = images.shape
    if height % grid_size or width % grid_size:
        raise ValueError("Image dimensions must be divisible by grid_size.")
    block_height = height // grid_size
    block_width = width // grid_size
    blocks = (
        images.view(batch_size, channels, grid_size, block_height, grid_size, block_width)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch_size, grid_size**2, channels * block_height * block_width)
    )
    permuted = torch.einsum("ij,bjk->bik", matrix, blocks)
    return (
        permuted.view(
            batch_size, grid_size, grid_size, channels, block_height, block_width
        )
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch_size, channels, height, width)
    )


def total_variation(images: Tensor) -> Tensor:
    vertical = (images[:, :, 1:] - images[:, :, :-1]).abs().mean()
    horizontal = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean()
    return vertical + horizontal


def adaptive_full_access_inversion(
    logits_fn: Callable[[Tensor], Tensor],
    *,
    target_class: int,
    image_shape: tuple[int, int, int],
    grid_size: int = 8,
    iterations: int = 400,
    restarts: int = 3,
    learning_rate: float = 0.05,
    tv_weight: float = 5e-4,
    l2_weight: float = 2e-3,
    entropy_weight: float = 0.05,
    sinkhorn_iterations: int = 10,
    minimum_iterations: int = 100,
    success_probability: float = 0.90,
    success_patience: int = 60,
    temperature_start: float = 1.0,
    temperature_end: float = 0.05,
    device: torch.device | None = None,
) -> InversionResult:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_count = grid_size**2
    target = torch.tensor([target_class], device=device)
    best: InversionResult | None = None

    for restart_index in range(restarts):
        image_parameter = torch.randn((1, *image_shape), device=device, requires_grad=True)
        permutation_logits = torch.randn(
            (block_count, block_count), device=device, requires_grad=True
        )
        optimizer = torch.optim.Adam(
            [image_parameter, permutation_logits], lr=learning_rate
        )
        successful_steps = 0
        completed = 0

        for step in range(iterations):
            completed = step + 1
            progress = step / max(iterations - 1, 1)
            temperature = temperature_start * (
                temperature_end / temperature_start
            ) ** progress
            optimizer.zero_grad(set_to_none=True)
            image = image_parameter.sigmoid()
            permutation = sinkhorn(
                permutation_logits, sinkhorn_iterations, temperature
            )
            candidate = soft_permute_blocks(image, permutation, grid_size)
            logits = logits_fn(candidate)
            permutation_entropy = -(
                permutation * permutation.clamp_min(1e-12).log()
            ).sum() / block_count
            loss = (
                torch.nn.functional.cross_entropy(logits, target)
                + tv_weight * total_variation(image)
                + l2_weight * image.square().mean()
                + entropy_weight * permutation_entropy
            )
            loss.backward()
            optimizer.step()

            probability = float(torch.softmax(logits.detach(), dim=1)[0, target_class])
            successful_steps = successful_steps + 1 if probability >= success_probability else 0
            if completed >= minimum_iterations and successful_steps >= success_patience:
                break

        with torch.no_grad():
            image = image_parameter.sigmoid()
            permutation = sinkhorn(
                permutation_logits, sinkhorn_iterations, temperature_end
            )
            candidate = soft_permute_blocks(image, permutation, grid_size)
            probabilities = torch.softmax(logits_fn(candidate), dim=1)
            target_probability = float(probabilities[0, target_class])
            result = InversionResult(
                image=candidate.detach().cpu(),
                target_probability=target_probability,
                target_success=int(probabilities.argmax(dim=1)) == target_class,
                objective=float(-torch.log(probabilities[0, target_class].clamp_min(1e-12))),
                restart_index=restart_index,
                iterations_run=completed,
            )
            if best is None or result.target_probability > best.target_probability:
                best = result

    if best is None:
        raise RuntimeError("No inversion restart was executed.")
    return best
