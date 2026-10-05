from __future__ import annotations

import torch

from label_noise_shape import fit_geometry, generate_margin_noise, generate_proto_noise


def make_geometry() -> tuple[torch.Tensor, object]:
    generator = torch.Generator().manual_seed(19)
    centers = torch.tensor(
        [
            [1.0, 0.5, -0.2, 0.0],
            [0.7, 0.8, -0.1, 0.1],
            [-0.5, 0.1, 0.9, -0.2],
        ]
    )
    labels = torch.arange(3).repeat_interleave(30)
    embeddings = centers[labels] + 0.4 * torch.randn(
        (labels.numel(), centers.shape[1]), generator=generator
    )
    return labels, fit_geometry(embeddings, labels)


def test_proto_noise_has_exact_budget_and_no_self_flips() -> None:
    labels, geometry = make_geometry()
    result = generate_proto_noise(labels, geometry, budget=21, beta=2.5, seed=3)

    assert result.flip_count == 21
    assert result.selected_indices.unique().numel() == 21
    assert torch.equal(result.flipped_mask, result.noisy_labels != labels)
    assert bool(
        (
            result.noisy_labels[result.selected_indices]
            != labels[result.selected_indices]
        ).all()
    )
    assert torch.allclose(result.destination_probabilities.sum(dim=1), torch.ones(3))
    assert torch.equal(result.destination_probabilities.diag(), torch.zeros(3))
    assert result.intended_transition is not None
    assert torch.allclose(result.intended_transition.sum(dim=1), torch.ones(3))


def test_margin_noise_selects_smallest_true_class_margins() -> None:
    labels, geometry = make_geometry()
    result = generate_margin_noise(labels, geometry, budget=17, beta=2.5, seed=4)

    assert result.flip_count == 17
    assert result.margins is not None
    expected = torch.topk(-result.margins, k=17).indices
    assert torch.equal(result.selected_indices, expected)
    assert bool((result.noisy_labels[expected] != labels[expected]).all())
    assert torch.allclose(
        result.destination_probabilities.sum(dim=1), torch.ones(labels.numel())
    )
    assert torch.equal(
        result.destination_probabilities[torch.arange(labels.numel()), labels],
        torch.zeros(labels.numel()),
    )


def test_generators_are_reproducible_for_fixed_seed() -> None:
    labels, geometry = make_geometry()
    first = generate_proto_noise(labels, geometry, budget=11, seed=8)
    second = generate_proto_noise(labels, geometry, budget=11, seed=8)
    assert torch.equal(first.noisy_labels, second.noisy_labels)

    first_margin = generate_margin_noise(labels, geometry, budget=11, seed=8)
    second_margin = generate_margin_noise(labels, geometry, budget=11, seed=8)
    assert torch.equal(first_margin.noisy_labels, second_margin.noisy_labels)
