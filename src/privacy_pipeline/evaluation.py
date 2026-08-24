"""Evaluation metrics used by the reported experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from skimage.metrics import structural_similarity
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass
class ClassificationMetrics:
    accuracy: float
    ece: float
    nll: float


@dataclass
class MembershipMetrics:
    oriented_auc: float
    effective_auc: float


@dataclass
class VerificationMetrics:
    eer: float
    tar_at_far: dict[float, float]


def _forward(model: nn.Module, images: Tensor, permutation_indices: Tensor) -> Tensor:
    return model(images, None if torch.all(permutation_indices < 0) else permutation_indices)


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    all_logits: list[Tensor] = []
    all_targets: list[Tensor] = []

    for images, targets, permutation_indices in loader:
        logits = _forward(model, images.to(device), permutation_indices.to(device))
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())
    return torch.cat(all_logits), torch.cat(all_targets)


def expected_calibration_error(
    probabilities: Tensor, labels: Tensor, bins: int = 15
) -> float:
    confidences, predictions = probabilities.max(dim=1)
    boundaries = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    ece = torch.zeros((), device=probabilities.device)

    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        if in_bin.any():
            accuracy = predictions[in_bin].eq(labels[in_bin]).float().mean()
            mean_confidence = confidences[in_bin].mean()
            ece += in_bin.float().mean() * (accuracy - mean_confidence).abs()
    return float(ece)


def classification_metrics(
    logits: Tensor, soft_targets: Tensor, temperature: float = 1.0
) -> ClassificationMetrics:
    labels = soft_targets.argmax(dim=1)
    scaled_logits = logits / temperature
    probabilities = torch.softmax(scaled_logits, dim=1)
    accuracy = float(probabilities.argmax(dim=1).eq(labels).float().mean() * 100)
    nll = float(torch.nn.functional.cross_entropy(scaled_logits, labels))
    ece = expected_calibration_error(probabilities, labels)
    return ClassificationMetrics(accuracy=accuracy, ece=ece, nll=nll)


def fit_temperature(logits: Tensor, soft_targets: Tensor) -> float:
    labels = soft_targets.argmax(dim=1)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=100)

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20)
        loss = torch.nn.functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20))


def membership_metrics(
    member_scores: Sequence[float], nonmember_scores: Sequence[float]
) -> MembershipMetrics:
    labels = np.concatenate(
        [np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    )
    scores = np.concatenate([member_scores, nonmember_scores])
    oriented_auc = float(roc_auc_score(labels, scores))
    return MembershipMetrics(
        oriented_auc=oriented_auc,
        effective_auc=max(oriented_auc, 1.0 - oriented_auc),
    )


def verification_metrics(
    embeddings: Tensor,
    labels: Tensor,
    fars: Iterable[float] = (1e-2, 1e-3),
) -> VerificationMetrics:
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    similarities = normalized @ normalized.T
    same_identity = labels[:, None].eq(labels[None, :])
    upper = torch.triu(torch.ones_like(same_identity, dtype=torch.bool), diagonal=1)
    pair_scores = similarities[upper].cpu().numpy()
    pair_labels = same_identity[upper].cpu().numpy().astype(int)
    fpr, tpr, _ = roc_curve(pair_labels, pair_scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fnr[eer_index] + fpr[eer_index]) / 2)

    tar_at_far: dict[float, float] = {}
    for far in fars:
        valid = np.where(fpr <= far)[0]
        tar_at_far[float(far)] = float(tpr[valid[-1]]) if len(valid) else 0.0
    return VerificationMetrics(eer=eer, tar_at_far=tar_at_far)


def reconstruction_metrics(
    reconstructed: Tensor, references: Tensor
) -> dict[str, float]:
    reconstructed = reconstructed.detach().cpu().clamp(0, 1)
    references = references.detach().cpu().clamp(0, 1)
    mse = torch.mean((reconstructed - references) ** 2).item()
    ssim_values: list[float] = []

    for reconstructed_image, reference_image in zip(reconstructed, references):
        ssim_values.append(
            structural_similarity(
                reference_image.permute(1, 2, 0).numpy(),
                reconstructed_image.permute(1, 2, 0).numpy(),
                channel_axis=2,
                data_range=1.0,
            )
        )
    return {"mse": mse, "ssim": float(np.mean(ssim_values))}
