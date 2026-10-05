from __future__ import annotations

import torch

from label_noise_shape import fit_geometry, generate_margin_noise, generate_proto_noise


def main() -> None:
    generator = torch.Generator().manual_seed(7)
    class_centers = torch.tensor(
        [
            [1.0, 0.4, -0.2, 0.1],
            [0.8, 0.6, -0.1, 0.0],
            [-0.6, 0.2, 0.8, -0.1],
        ]
    )
    labels = torch.arange(3).repeat_interleave(20)
    embeddings = class_centers[labels] + 0.35 * torch.randn(
        (labels.numel(), class_centers.shape[1]), generator=generator
    )
    geometry = fit_geometry(embeddings, labels)

    for name, result in (
        (
            "ProtoNoise",
            generate_proto_noise(labels, geometry, budget=12, beta=2.5, seed=0),
        ),
        (
            "MarginNoise",
            generate_margin_noise(labels, geometry, budget=12, beta=2.5, seed=0),
        ),
    ):
        print(f"\n{name}: {result.flip_count} exact flips")
        print(result.empirical_transition.numpy().round(3))


if __name__ == "__main__":
    main()
