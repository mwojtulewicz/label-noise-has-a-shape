from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import Geometry, true_class_margin
from .metrics import empirical_transition_matrix


@dataclass(frozen=True)
class NoiseResult:
    clean_labels: torch.Tensor
    noisy_labels: torch.Tensor
    flipped_mask: torch.Tensor
    selected_indices: torch.Tensor
    destination_probabilities: torch.Tensor
    empirical_transition: torch.Tensor
    intended_transition: torch.Tensor | None = None
    margins: torch.Tensor | None = None

    @property
    def flip_count(self) -> int:
        return int(self.flipped_mask.sum().item())


def _validate_inputs(
    labels: torch.Tensor,
    geometry: Geometry,
    budget: int,
    beta: float,
) -> tuple[torch.Tensor, int, int]:
    labels = torch.as_tensor(labels, dtype=torch.long).flatten().cpu()
    sample_count, num_classes = geometry.instance_similarity.shape
    if labels.numel() != sample_count:
        raise ValueError("labels and geometry must use the same sample order")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError("labels contain an out-of-range class")
    budget = int(budget)
    if budget < 0 or budget > sample_count:
        raise ValueError("budget must lie between zero and the sample count")
    if beta <= 0:
        raise ValueError("beta must be positive")
    return labels, budget, num_classes


def _wrong_class_probabilities(
    scores: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    masked = scores.clone()
    masked[torch.arange(scores.shape[0]), labels] = -torch.inf
    return torch.softmax(float(beta) * masked, dim=1)


def _class_destination_probabilities(
    class_similarity: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    masked = class_similarity.clone()
    masked.fill_diagonal_(-torch.inf)
    return torch.softmax(float(beta) * masked, dim=1)


def generate_proto_noise(
    labels: torch.Tensor,
    geometry: Geometry,
    *,
    budget: int,
    beta: float = 2.5,
    seed: int = 0,
) -> NoiseResult:
    """Generate ProtoNoise with an exact global flip budget.

    Examples are sampled uniformly without replacement. Their destination
    labels are sampled from a softmax over wrong-class prototype similarities.
    """

    labels, budget, num_classes = _validate_inputs(labels, geometry, budget, beta)
    class_probs = _class_destination_probabilities(
        geometry.class_similarity, beta
    ).cpu()
    generator = torch.Generator().manual_seed(int(seed))

    # The paper implementation sampled from constant float64 weights. This is
    # uniform without replacement while preserving the exact RNG behavior.
    weights = torch.ones(labels.numel(), dtype=torch.float64)
    selected = (
        torch.multinomial(weights, budget, replacement=False, generator=generator)
        if budget
        else torch.empty(0, dtype=torch.long)
    )
    noisy = labels.clone()
    if budget:
        noisy[selected] = torch.multinomial(
            class_probs[labels[selected]], 1, generator=generator
        ).squeeze(1)

    rate = budget / max(labels.numel(), 1)
    intended = class_probs * rate
    intended[torch.arange(num_classes), torch.arange(num_classes)] = 1.0 - rate
    flipped = noisy != labels
    return NoiseResult(
        clean_labels=labels,
        noisy_labels=noisy,
        flipped_mask=flipped,
        selected_indices=selected,
        destination_probabilities=class_probs,
        intended_transition=intended,
        empirical_transition=empirical_transition_matrix(labels, noisy, num_classes),
    )


def generate_margin_noise(
    labels: torch.Tensor,
    geometry: Geometry,
    *,
    budget: int,
    beta: float = 2.5,
    seed: int = 0,
) -> NoiseResult:
    """Generate MarginNoise with an exact global flip budget.

    The selected examples have the smallest signed margin
    ``similarity(true class) - max similarity(wrong class)``. Destinations are
    sampled from each selected example's wrong-class similarity softmax.
    """

    labels, budget, num_classes = _validate_inputs(labels, geometry, budget, beta)
    scores = geometry.instance_similarity.cpu()
    margins = true_class_margin(scores, labels)
    probabilities = _wrong_class_probabilities(scores, labels, beta)
    selected = (
        torch.topk(-margins, k=budget).indices
        if budget
        else torch.empty(0, dtype=torch.long)
    )

    generator = torch.Generator().manual_seed(int(seed))
    noisy = labels.clone()
    if budget:
        noisy[selected] = torch.multinomial(
            probabilities[selected], 1, generator=generator
        ).squeeze(1)
    flipped = noisy != labels
    return NoiseResult(
        clean_labels=labels,
        noisy_labels=noisy,
        flipped_mask=flipped,
        selected_indices=selected,
        destination_probabilities=probabilities,
        margins=margins,
        empirical_transition=empirical_transition_matrix(labels, noisy, num_classes),
    )
