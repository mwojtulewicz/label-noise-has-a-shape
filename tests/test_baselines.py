from __future__ import annotations

import torch

from label_noise_shape import (
    CIFAR10_SEMANTIC2_TARGETS,
    generate_semantic2_noise,
    generate_symmetric_noise,
    generate_xia_idln_noise,
)


def test_symmetric_noise_has_exact_budget_and_is_reproducible() -> None:
    labels = torch.arange(5).repeat_interleave(20)
    first = generate_symmetric_noise(labels, budget=27, num_classes=5, seed=12)
    second = generate_symmetric_noise(labels, budget=27, num_classes=5, seed=12)

    assert first.flip_count == 27
    assert torch.equal(first.noisy_labels, second.noisy_labels)
    assert bool(
        (
            first.noisy_labels[first.selected_indices] != labels[first.selected_indices]
        ).all()
    )
    assert torch.allclose(
        first.destination_probabilities.sum(dim=1), torch.ones(5, dtype=torch.float64)
    )


def test_semantic2_uses_only_declared_balanced_edges() -> None:
    labels = torch.arange(10).repeat_interleave(30)
    result = generate_semantic2_noise(labels, budget=91, seed=4)

    assert result.flip_count == 91
    counts = torch.zeros((10, 10), dtype=torch.long)
    counts.index_put_(
        (labels, result.noisy_labels), torch.ones_like(labels), accumulate=True
    )
    edge_counts = counts.clone()
    edge_counts.fill_diagonal_(0)
    assert int(edge_counts.sum()) == 91
    for source, targets in enumerate(CIFAR10_SEMANTIC2_TARGETS):
        used = set(torch.where(edge_counts[source] > 0)[0].tolist())
        assert used.issubset(set(targets))
        assert (
            abs(
                int(edge_counts[source, targets[0]])
                - int(edge_counts[source, targets[1]])
            )
            <= 1
        )
    row_flips = edge_counts.sum(dim=1)
    # Each of the 20 directed edge counts differs by at most one. A source row
    # contains two edges, so their sums can differ by at most two.
    assert (
        int(edge_counts[edge_counts > 0].max() - edge_counts[edge_counts > 0].min())
        <= 1
    )
    assert int(row_flips.max() - row_flips.min()) <= 2


def test_xia_idln_has_exact_budget_no_self_flips_and_fixed_seed() -> None:
    generator = torch.Generator().manual_seed(2)
    inputs = torch.randint(
        0, 256, (80, 3, 4, 4), generator=generator, dtype=torch.uint8
    )
    labels = torch.arange(4).repeat_interleave(20)
    first = generate_xia_idln_noise(inputs, labels, budget=29, num_classes=4, seed=7)
    second = generate_xia_idln_noise(inputs, labels, budget=29, num_classes=4, seed=7)

    assert first.flip_count == 29
    assert torch.equal(first.selected_indices, second.selected_indices)
    assert torch.equal(first.noisy_labels, second.noisy_labels)
    assert bool(
        (
            first.noisy_labels[first.selected_indices] != labels[first.selected_indices]
        ).all()
    )
    assert first.destination_probabilities.shape == (29, 4)
    assert torch.allclose(first.destination_probabilities.sum(dim=1), torch.ones(29))
    assert torch.equal(
        first.destination_probabilities[
            torch.arange(29), labels[first.selected_indices]
        ],
        torch.zeros(29),
    )
