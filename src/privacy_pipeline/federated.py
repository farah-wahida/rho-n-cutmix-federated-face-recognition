"""Federated training utilities for the privacy pipeline."""

from __future__ import annotations

import math
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class RoundMetrics:
    round_index: int
    mean_client_loss: float
    participating_clients: int
    learning_rate: float
    validation_loss: float | None = None


def soft_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def confidence_penalty(logits: Tensor) -> Tensor:
    probabilities = torch.softmax(logits, dim=1)
    return (probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1).mean()


def _forward(model: nn.Module, images: Tensor, permutation_indices: Tensor) -> Tensor:
    return model(images, None if torch.all(permutation_indices < 0) else permutation_indices)


def cosine_learning_rate(
    round_index: int,
    rounds: int,
    initial: float,
    minimum: float,
) -> float:
    progress = (round_index - 1) / max(rounds - 1, 1)
    return minimum + 0.5 * (initial - minimum) * (1.0 + math.cos(math.pi * progress))


def train_client(
    global_model: nn.Module,
    loader: DataLoader,
    *,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    confidence_penalty_weight: float,
    gradient_clip_norm: float = 5.0,
    device: torch.device | None = None,
) -> tuple[OrderedDict[str, Tensor], float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deepcopy(global_model).to(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_loss = 0.0
    total_samples = 0

    for _ in range(local_epochs):
        for images, targets, permutation_indices in loader:
            images = images.to(device)
            targets = targets.to(device)
            permutation_indices = permutation_indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, images, permutation_indices)
            loss = soft_cross_entropy(logits, targets)
            if confidence_penalty_weight > 0:
                loss = loss + confidence_penalty_weight * confidence_penalty(logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach()) * images.shape[0]
            total_samples += images.shape[0]

    state = OrderedDict(
        (name, value.detach().cpu()) for name, value in model.state_dict().items()
    )
    return state, total_loss / max(total_samples, 1)


def fedavg(
    client_states: Sequence[OrderedDict[str, Tensor]],
    sample_counts: Sequence[int],
) -> OrderedDict[str, Tensor]:
    if not client_states:
        raise ValueError("At least one client state is required.")
    if len(client_states) != len(sample_counts):
        raise ValueError("Client states and sample counts must have the same length.")
    total_samples = float(sum(sample_counts))
    if total_samples <= 0:
        raise ValueError("The total client sample count must be positive.")

    averaged: OrderedDict[str, Tensor] = OrderedDict()
    for name in client_states[0]:
        reference = client_states[0][name]
        if not reference.is_floating_point():
            averaged[name] = reference.clone()
            continue
        value = torch.zeros_like(reference)
        for state, count in zip(client_states, sample_counts):
            value.add_(state[name], alpha=count / total_samples)
        averaged[name] = value
    return averaged


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for images, targets, permutation_indices in loader:
        images = images.to(device)
        targets = targets.to(device)
        permutation_indices = permutation_indices.to(device)
        loss = soft_cross_entropy(_forward(model, images, permutation_indices), targets)
        total_loss += float(loss) * images.shape[0]
        total_samples += images.shape[0]
    return total_loss / max(total_samples, 1)


def federated_train(
    model: nn.Module,
    dataset: Dataset,
    client_indices: Sequence[Sequence[int]],
    *,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    minimum_learning_rate: float = 0.0,
    weight_decay: float = 0.0,
    gradient_clip_norm: float = 5.0,
    confidence_penalty_weight: float = 0.0,
    validation_loader: DataLoader | None = None,
    early_stopping_patience: int = 0,
    device: torch.device | None = None,
) -> list[RoundMetrics]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history: list[RoundMetrics] = []
    best_loss = float("inf")
    best_state: OrderedDict[str, Tensor] | None = None
    stale_rounds = 0

    for round_index in range(1, rounds + 1):
        round_lr = cosine_learning_rate(
            round_index, rounds, learning_rate, minimum_learning_rate
        )
        client_states: list[OrderedDict[str, Tensor]] = []
        sample_counts: list[int] = []
        client_losses: list[float] = []

        for indices in client_indices:
            if not indices:
                continue
            loader = DataLoader(
                Subset(dataset, list(indices)), batch_size=batch_size, shuffle=True
            )
            state, loss = train_client(
                model,
                loader,
                local_epochs=local_epochs,
                learning_rate=round_lr,
                weight_decay=weight_decay,
                confidence_penalty_weight=confidence_penalty_weight,
                gradient_clip_norm=gradient_clip_norm,
                device=device,
            )
            client_states.append(state)
            sample_counts.append(len(indices))
            client_losses.append(loss)

        model.load_state_dict(fedavg(client_states, sample_counts))
        validation_loss = (
            evaluate_loss(model, validation_loader, device)
            if validation_loader is not None
            else None
        )
        history.append(
            RoundMetrics(
                round_index=round_index,
                mean_client_loss=sum(client_losses) / len(client_losses),
                participating_clients=len(client_states),
                learning_rate=round_lr,
                validation_loss=validation_loss,
            )
        )

        if validation_loss is not None and early_stopping_patience > 0:
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = OrderedDict(
                    (name, value.detach().cpu().clone())
                    for name, value in model.state_dict().items()
                )
                stale_rounds = 0
            else:
                stale_rounds += 1
                if stale_rounds >= early_stopping_patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
