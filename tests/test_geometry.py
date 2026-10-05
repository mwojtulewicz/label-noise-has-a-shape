from __future__ import annotations

import torch

from label_noise_shape import fit_geometry, true_class_margin


def test_fit_geometry_centers_and_normalizes() -> None:
    embeddings = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [2.5, 1.2, 0.1],
            [0.0, 2.0, 2.0],
            [0.2, 2.3, 1.8],
            [-2.0, -1.0, 1.0],
            [-2.2, -0.7, 1.3],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    geometry = fit_geometry(embeddings, labels)

    assert torch.allclose(
        geometry.centered_embeddings.mean(dim=0),
        torch.zeros(3),
        atol=1e-7,
    )
    assert torch.allclose(
        geometry.normalized_embeddings.norm(dim=1),
        torch.ones(6),
        atol=1e-6,
    )
    assert torch.allclose(
        geometry.prototypes.norm(dim=1),
        torch.ones(3),
        atol=1e-6,
    )
    assert torch.allclose(
        geometry.class_similarity,
        geometry.class_similarity.T,
        atol=1e-7,
    )


def test_true_class_margin_is_signed_label_conditioned_gap() -> None:
    scores = torch.tensor(
        [
            [0.8, 0.7, 0.1],
            [0.6, 0.9, 0.2],
            [0.4, 0.5, 0.3],
        ]
    )
    labels = torch.tensor([0, 1, 2])

    margins = true_class_margin(scores, labels)

    assert torch.allclose(margins, torch.tensor([0.1, 0.3, -0.2]))
