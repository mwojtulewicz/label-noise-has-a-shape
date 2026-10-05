from __future__ import annotations

import torch


def empirical_transition_matrix(
    clean_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
    num_classes: int | None = None,
) -> torch.Tensor:
    clean = torch.as_tensor(clean_labels, dtype=torch.long).flatten().cpu()
    noisy = torch.as_tensor(noisy_labels, dtype=torch.long).flatten().cpu()
    if clean.numel() != noisy.numel():
        raise ValueError("clean and noisy labels must have equal length")
    inferred = (
        0 if not clean.numel() else int(torch.maximum(clean.max(), noisy.max())) + 1
    )
    num_classes = inferred if num_classes is None else int(num_classes)
    if num_classes < inferred:
        raise ValueError("num_classes does not cover all labels")

    counts = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    if clean.numel():
        flat = clean * num_classes + noisy
        counts = (
            torch.bincount(flat, minlength=num_classes**2)
            .reshape(num_classes, num_classes)
            .to(torch.float64)
        )
    return counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
