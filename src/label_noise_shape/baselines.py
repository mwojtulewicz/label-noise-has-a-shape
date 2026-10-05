from __future__ import annotations

from collections.abc import Sequence

import torch

from .generators import NoiseResult
from .metrics import empirical_transition_matrix

# CIFAR-10 order: airplane, automobile, bird, cat, deer, dog, frog,
# horse, ship, truck. Every class has in-degree and out-degree two.
CIFAR10_SEMANTIC2_TARGETS: tuple[tuple[int, int], ...] = (
    (2, 1),
    (9, 8),
    (0, 6),
    (5, 6),
    (7, 2),
    (3, 7),
    (4, 3),
    (4, 5),
    (0, 9),
    (1, 8),
)


def _validate_labels_and_budget(
    labels: torch.Tensor,
    budget: int,
    num_classes: int,
) -> tuple[torch.Tensor, int, int]:
    labels = torch.as_tensor(labels, dtype=torch.long).flatten().cpu()
    budget = int(budget)
    num_classes = int(num_classes)
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if budget < 0 or budget > labels.numel():
        raise ValueError("budget must lie between zero and the sample count")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError("labels contain an out-of-range class")
    return labels, budget, num_classes


def _sample_exact_indices(
    weights: torch.Tensor,
    budget: int,
    generator: torch.Generator,
) -> torch.Tensor:
    weights = torch.as_tensor(weights, dtype=torch.float64).flatten().clamp_min(0)
    if budget == 0:
        return torch.empty(0, dtype=torch.long)
    if int((weights > 0).sum().item()) < budget:
        weights = weights + (weights == 0).to(weights.dtype) * 1e-12
    return torch.multinomial(weights, budget, replacement=False, generator=generator)


def generate_symmetric_noise(
    labels: torch.Tensor,
    *,
    budget: int,
    num_classes: int,
    seed: int = 0,
) -> NoiseResult:
    """Generate exact-budget symmetric label noise.

    Examples are selected uniformly without replacement, and every selected
    label is replaced uniformly by one of the other classes.
    """

    labels, budget, num_classes = _validate_labels_and_budget(
        labels, budget, num_classes
    )
    generator = torch.Generator().manual_seed(int(seed))
    selected = torch.randperm(labels.numel(), generator=generator)[:budget]
    noisy = labels.clone()
    if budget:
        offsets = torch.randint(1, num_classes, (budget,), generator=generator)
        noisy[selected] = (labels[selected] + offsets) % num_classes

    class_probs = torch.full(
        (num_classes, num_classes),
        1.0 / (num_classes - 1),
        dtype=torch.float64,
    )
    class_probs.fill_diagonal_(0.0)
    rate = budget / max(labels.numel(), 1)
    intended = class_probs * rate
    intended.diagonal().copy_(torch.full((num_classes,), 1.0 - rate))
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


def _semantic_weights(
    targets: Sequence[Sequence[int]],
    num_classes: int,
) -> torch.Tensor:
    if len(targets) != num_classes:
        raise ValueError("semantic targets must define every class")
    width = len(targets[0]) if targets else 0
    if width == 0 or any(len(row) != width for row in targets):
        raise ValueError("every source class must have the same nonzero target count")

    weights = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    for source, row in enumerate(targets):
        normalized = [int(target) for target in row]
        if len(set(normalized)) != width:
            raise ValueError(f"source class {source} contains duplicate targets")
        for target in normalized:
            if target < 0 or target >= num_classes:
                raise ValueError(f"source class {source} has an out-of-range target")
            if target == source:
                raise ValueError(f"source class {source} targets itself")
            weights[source, target] = 1.0
    return weights


def _allocate_semantic_edge_counts(
    weights: torch.Tensor,
    class_counts: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    mass = class_counts.to(torch.float64)[:, None] * weights
    mass /= mass.sum()
    desired = mass * budget
    desired_rows = desired.sum(dim=1)
    if bool((desired_rows > class_counts.to(torch.float64) + 1e-9).any()):
        bad = torch.where(desired_rows > class_counts.to(torch.float64) + 1e-9)[0]
        raise ValueError(f"source-class capacity exceeded for rows {bad.tolist()}")

    allocated = torch.floor(desired).to(torch.long)
    remaining = budget - int(allocated.sum().item())
    fractions = (desired - allocated.to(desired.dtype)).flatten()
    order = torch.argsort(fractions, descending=True)
    row_used = allocated.sum(dim=1)
    width = weights.shape[1]
    for flat_index in order.tolist():
        if remaining == 0:
            break
        source, target = divmod(flat_index, width)
        if source == target or float(weights[source, target]) <= 0.0:
            continue
        if int(row_used[source]) >= int(class_counts[source]):
            continue
        allocated[source, target] += 1
        row_used[source] += 1
        remaining -= 1
    if remaining:
        raise ValueError(
            f"could not allocate {remaining} flips within class capacities"
        )
    return allocated


def _realize_edge_counts(
    labels: torch.Tensor,
    edge_counts: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    noisy = labels.clone()
    for source in range(edge_counts.shape[0]):
        source_indices = torch.where(labels == source)[0]
        order = source_indices[
            torch.randperm(source_indices.numel(), generator=generator)
        ]
        cursor = 0
        for target in range(edge_counts.shape[1]):
            if source == target:
                continue
            count = int(edge_counts[source, target])
            if count:
                noisy[order[cursor : cursor + count]] = target
                cursor += count
    return noisy


def generate_semantic2_noise(
    labels: torch.Tensor,
    *,
    budget: int,
    seed: int = 0,
    targets: Sequence[Sequence[int]] = CIFAR10_SEMANTIC2_TARGETS,
) -> NoiseResult:
    """Generate the paper's exact-budget, balanced Semantic-2 baseline.

    The default target graph is defined for CIFAR-10. Equal continuous mass is
    assigned to both destinations in each source row, then largest-remainder
    allocation realizes the requested global integer budget.
    """

    num_classes = len(targets)
    labels, budget, num_classes = _validate_labels_and_budget(
        labels, budget, num_classes
    )
    weights = _semantic_weights(targets, num_classes)
    class_counts = torch.bincount(labels, minlength=num_classes)
    edge_counts = _allocate_semantic_edge_counts(weights, class_counts, budget)
    noisy = _realize_edge_counts(labels, edge_counts, seed)
    selected = torch.where(noisy != labels)[0]

    class_probs = weights / weights.sum(dim=1, keepdim=True)
    intended = edge_counts.to(torch.float64) / class_counts[:, None].clamp_min(1)
    intended.diagonal().copy_(
        1.0 - edge_counts.sum(dim=1).to(torch.float64) / class_counts.clamp_min(1)
    )
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


def _sample_truncated_normal(
    count: int,
    mean: float,
    std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if not 0.0 <= mean <= 1.0:
        raise ValueError("truncated-normal mean must lie in [0, 1]")
    if std < 0.0:
        raise ValueError("rate_std must be nonnegative")
    if std == 0.0:
        return torch.full((count,), mean, dtype=torch.float64)

    values = torch.empty(count, dtype=torch.float64)
    remaining = torch.arange(count)
    while remaining.numel():
        draws = mean + std * torch.randn(
            remaining.numel(), generator=generator, dtype=torch.float64
        )
        accepted = (draws >= 0.0) & (draws <= 1.0)
        values[remaining[accepted]] = draws[accepted]
        remaining = remaining[~accepted]
    return values


def generate_xia_idln_noise(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    budget: int,
    num_classes: int,
    seed: int = 0,
    rate_std: float = 0.1,
    input_scale: float = 255.0,
) -> NoiseResult:
    """Generate the paper's exact-budget adaptation of Xia et al.'s IdLN.

    Algorithm 2 of Xia et al. draws truncated-normal instance propensities and
    class-specific Gaussian projections of raw inputs. Here, propensities weight
    sampling without replacement so exactly ``budget`` labels change. The
    returned destination probabilities correspond, in selected-index order, to
    the selected examples only.
    """

    labels, budget, num_classes = _validate_labels_and_budget(
        labels, budget, num_classes
    )
    raw = torch.as_tensor(inputs).cpu()
    if raw.ndim < 2 or raw.shape[0] != labels.numel():
        raise ValueError("inputs and labels must share their first dimension")
    if input_scale <= 0:
        raise ValueError("input_scale must be positive")

    generator = torch.Generator().manual_seed(int(seed))
    rate = budget / max(labels.numel(), 1)
    propensities = _sample_truncated_normal(
        labels.numel(), rate, float(rate_std), generator
    )
    selected = _sample_exact_indices(propensities, budget, generator)
    selected_probabilities = torch.empty((budget, num_classes), dtype=torch.float32)
    noisy = labels.clone()
    feature_dim = int(raw[0].numel()) if raw.shape[0] else 0

    for source in range(num_classes):
        # Drawing projections for every class keeps seeded results independent
        # of whether a source class happens to occur in the selected subset.
        projection = torch.randn(
            (feature_dim, num_classes), generator=generator, dtype=torch.float32
        )
        positions = torch.where(labels[selected] == source)[0]
        if not positions.numel():
            continue
        source_indices = selected[positions]
        features = raw[source_indices].reshape(source_indices.numel(), -1).float()
        logits = (features / float(input_scale)) @ projection
        logits[:, source] = -torch.inf
        probabilities = torch.softmax(logits, dim=1)
        selected_probabilities[positions] = probabilities
        noisy[source_indices] = torch.multinomial(
            probabilities, 1, generator=generator
        ).squeeze(1)

    flipped = noisy != labels
    return NoiseResult(
        clean_labels=labels,
        noisy_labels=noisy,
        flipped_mask=flipped,
        selected_indices=selected,
        destination_probabilities=selected_probabilities,
        empirical_transition=empirical_transition_matrix(labels, noisy, num_classes),
    )
