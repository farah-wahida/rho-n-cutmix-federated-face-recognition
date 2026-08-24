"""Federated training utilities for the privacy pipeline."""

from __future__ import annotations

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


def soft_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def confidence_penalty(logits: Tensor) -> Tensor:
    probabilities = torch.softmax(logits, dim=1)
    return (probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1).mean()


def _forward(model: nn.Module, images: Tensor, permutation_indices: Tensor) -> Tensor:
    return model(images, None if torch.all(permutation_indices < 0) else permutation_indices)


def train_client(
    global_model: nn.Module,
    loader: DataLoader,
    *,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    confidence_penalty_weight: float,
    device: torch.device,
) -> tuple[OrderedDict[str, Tensor], float]:
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


def federated_train(
    model: nn.Module,
    dataset: Dataset,
    client_indices: Sequence[Sequence[int]],
    *,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    confidence_penalty_weight: float = 0.0,
    device: torch.device | None = None,
) -> list[RoundMetrics]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history: list[RoundMetrics] = []

    for round_index in range(1, rounds + 1):
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
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                confidence_penalty_weight=confidence_penalty_weight,
                device=device,
            )
            client_states.append(state)
            sample_counts.append(len(indices))
            client_losses.append(loss)

        model.load_state_dict(fedavg(client_states, sample_counts))
        history.append(
            RoundMetrics(
                round_index=round_index,
                mean_client_loss=sum(client_losses) / len(client_losses),
                participating_clients=len(client_states),
            )
        )
    return history
