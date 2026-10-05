# Label Noise Has a Shape

[![Tests](https://github.com/mwojtulewicz/label-noise-has-a-shape/actions/workflows/tests.yml/badge.svg)](https://github.com/mwojtulewicz/label-noise-has-a-shape/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Minimal official implementation and reproducibility package for:

> Mateusz Wojtulewicz, Piotr Duda, and Leszek Rutkowski,
> **Label Noise Has a Shape: Geometry-Guided Benchmark Generation for
> Noisy-Label Learning**, OWAD at IEEE ICDM Workshops, 2026.

The repository contains the paper's two proposed exact-budget benchmark
generators:

- **ProtoNoise** selects examples uniformly and directs their corrupted labels
  toward classes with similar embedding-space prototypes.
- **MarginNoise** spends the corruption budget on examples with the smallest
  signed true-class prototype margins, then samples their destinations from
  instance-to-prototype similarities.

Both methods use a frozen, dataset-mean-centered representation. The paper's
CIFAR-10 experiments use a 128-dimensional signed bottleneck from a
CIFAR-adapted ResNet-18-like autoencoder and concentration `beta = 2.5`.

For matched-budget comparisons, the package also implements the three
synthetic baselines used in the paper: exact-budget symmetric noise, balanced
Semantic-2 noise, and the exact-budget adaptation of Xia et al.'s IdLN. Human
errors are read from CIFAR-10N and are not synthetically generated.

## Quick start

```bash
git clone https://github.com/mwojtulewicz/label-noise-has-a-shape.git
cd label-noise-has-a-shape
uv sync --extra dev
uv run pytest
uv run python scripts/demo.py
uv run python scripts/check_reference.py
```

The demo uses synthetic embeddings and finishes on CPU. It verifies the exact
flip budget and prints empirical transition matrices for both generators.

## Generate the paper's CIFAR-10 label sets

Train the frozen autoencoder:

```bash
uv run python scripts/train_autoencoder.py \
  --data-root data \
  --output checkpoints/cifar10_resnet18_ae_128d.pt
```

Generate ProtoNoise and MarginNoise labels at the CIFAR-10N Aggregate, Random
2, and Worst budgets:

```bash
uv run python scripts/generate_cifar10.py \
  --checkpoint checkpoints/cifar10_resnet18_ae_128d.pt \
  --data-root data \
  --output-dir outputs/paper_labels
```

Generate the symmetric, Semantic-2, and IdLN baseline labels at the same
budgets:

```bash
uv run python scripts/generate_baselines.py \
  --data-root data \
  --output-dir outputs/baseline_labels
```

Use `--methods symmetric semantic2` to omit the more computationally expensive
raw-image IdLN generator.

The selected paper checkpoint has SHA-256 digest
`8bf4568574185265482c9dd03e37b84350298e8bcdc4bf405c7815cb31cc79a5`.
Because training a neural autoencoder can vary slightly across hardware and
software stacks, an exact label-level reproduction should use that checkpoint
when it is attached to the corresponding GitHub release.

## Repository structure

| Path | Purpose |
|---|---|
| `src/label_noise_shape/` | Small importable implementation of the proposed and baseline generators |
| `scripts/` | CPU demo and CIFAR-10 training/generation entry points |
| `configs/` | Human-readable paper configuration and exact flip budgets |
| `results/reference/` | Compact reference summaries, never checkpoints or raw datasets |
| `docs/` | Experiment mapping, reproducibility boundaries, and provenance |
| `tests/` | Unit and exact-budget behavior tests |

Generated datasets, embeddings, checkpoints, labels, and experiment logs are
ignored by Git.

## Reproducibility scope

This compact release reproduces benchmark construction, including the
representation, centering, prototypes, margins, exact-budget selection, and
destination sampling. The paper evaluates the resulting labels with standard
and externally published robust learners. Their exact recipes and upstream
implementations are mapped in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md); bulky
third-party repositories are not vendored here.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before comparing newly
generated labels or downstream accuracies with the paper.

## Citation

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).
The final proceedings DOI and BibTeX entry will be added when available.

## License

Original code in this repository is released under the [MIT License](LICENSE).
Downloaded datasets, pretrained weights, and external learner implementations
remain subject to their respective licenses and citation requirements.
