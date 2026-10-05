from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Geometry:
    """Dataset-centered geometry used by ProtoNoise and MarginNoise."""

    centered_embeddings: torch.Tensor
    embedding_mean: torch.Tensor
    normalized_embeddings: torch.Tensor
    prototypes: torch.Tensor
    class_similarity: torch.Tensor
    instance_similarity: torch.Tensor


def _as_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    embeddings = torch.as_tensor(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must have non-empty shape (N, D)")
    if not embeddings.is_floating_point():
        embeddings = embeddings.float()
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings contain non-finite values")
    return embeddings.cpu()


def _as_labels(labels: torch.Tensor, sample_count: int) -> torch.Tensor:
    labels = torch.as_tensor(labels, dtype=torch.long).flatten().cpu()
    if labels.numel() != sample_count:
        raise ValueError("labels must have shape (N,) matching embeddings")
    if labels.numel() and int(labels.min()) < 0:
        raise ValueError("labels must be nonnegative")
    return labels


def normalize_rows(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norms = values.norm(p=2, dim=1, keepdim=True)
    if bool((norms <= eps).any()):
        raise ValueError("cannot L2-normalize a zero or near-zero row")
    return values / norms


def true_class_margin(
    instance_similarity: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return similarity(true class) minus the largest wrong-class similarity."""

    scores = torch.as_tensor(instance_similarity)
    labels = torch.as_tensor(labels, dtype=torch.long, device=scores.device).flatten()
    if scores.ndim != 2 or labels.numel() != scores.shape[0]:
        raise ValueError("scores and labels must have shapes (N, C) and (N,)")
    rows = torch.arange(scores.shape[0], device=scores.device)
    true_scores = scores[rows, labels]
    wrong_scores = scores.clone()
    wrong_scores[rows, labels] = -torch.inf
    return true_scores - wrong_scores.max(dim=1).values


def fit_geometry(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int | None = None,
) -> Geometry:
    """Fit the paper's centered, normalized one-prototype-per-class geometry.

    The training-set mean is subtracted before any cosine geometry is formed.
    Each centered example is L2-normalized, class prototypes are means of those
    normalized examples, and the prototypes are normalized once more.
    """

    embeddings = _as_embeddings(embeddings)
    labels = _as_labels(labels, embeddings.shape[0])
    inferred_classes = int(labels.max()) + 1 if labels.numel() else 0
    num_classes = inferred_classes if num_classes is None else int(num_classes)
    if num_classes < inferred_classes or num_classes < 2:
        raise ValueError("num_classes must cover all labels and be at least two")

    embedding_mean = embeddings.mean(dim=0, keepdim=True)
    centered = embeddings - embedding_mean
    normalized = normalize_rows(centered)

    prototypes = []
    for class_id in range(num_classes):
        class_mask = labels == class_id
        if not bool(class_mask.any()):
            raise ValueError(f"class {class_id} has no examples")
        prototypes.append(normalized[class_mask].mean(dim=0))
    prototypes_t = normalize_rows(torch.stack(prototypes, dim=0))

    class_similarity = prototypes_t @ prototypes_t.T
    instance_similarity = normalized @ prototypes_t.T
    return Geometry(
        centered_embeddings=centered,
        embedding_mean=embedding_mean.squeeze(0),
        normalized_embeddings=normalized,
        prototypes=prototypes_t,
        class_similarity=class_similarity,
        instance_similarity=instance_similarity,
    )
