# Paper experiment map

The paper compares six matched-budget label sets on the same CIFAR-10 training
images: symmetric noise, Semantic-2, the exact-budget IdLN adaptation of Xia et
al., CIFAR-10N human errors, ProtoNoise, and MarginNoise.

The two proposed generators and the three synthetic baselines use the
implementation in `src/label_noise_shape/`. `scripts/generate_baselines.py`
generates exact-budget symmetric, Semantic-2, and IdLN labels; the IdLN code is
the paper's stated exact-budget adaptation of Algorithm 2 from Xia et al. Human
labels must be obtained from the official CIFAR-10N release. The public release
does not vendor external learner repositories.

## Synthetic baseline definitions

- **Symmetric:** select the exact integer budget uniformly without replacement,
  then draw uniformly from the nine wrong labels.
- **Semantic-2:** assign each CIFAR-10 class two predeclared semantic
  destinations and balance directed-edge counts up to integer rounding. The
  complete target graph is exposed as `CIFAR10_SEMANTIC2_TARGETS`.
- **IdLN:** draw truncated-normal instance propensities and class-specific
  Gaussian raw-input projections as in Xia et al. The paper replaces the
  original Bernoulli selection with propensity-weighted sampling without
  replacement to satisfy the shared exact-budget protocol.

## Learner recipes

| Learner | Paper recipe | Seeds |
|---|---|---:|
| Cross entropy | CIFAR ResNet-18, SGD, cosine decay, 120 epochs | 0, 1, 2 |
| GCE | Same victim recipe, generalized cross-entropy loss | 0, 1, 2 |
| Co-teaching | Two-network sample-selection recipe, cosine decay, 120 epochs | 0, 1, 2 |
| SNV-F | One-stage forward correction, SNV transition estimate with `k=20` | 0, 1, 2 |
| ProMix | Official implementation scaled to 300 epochs | 0, 1 |

The common conventional victim uses initial learning rate `0.1`, momentum
`0.9`, weight decay `5e-4`, standard random crop and horizontal flip, and cosine
decay. ProMix uses batch size 128, initial learning rate `0.05`, 5 pretraining
epochs, a 25-epoch ramp, and label expansion over the final 125 epochs.

## Evidence boundary

The camera-ready accuracy table is an aggregation of multiple GPU runs, not a
single command. The public release will include a compact table-level CSV and
validation script, while raw checkpoints and complete scheduler logs remain
outside Git. This keeps the repository small without obscuring seed coverage or
paper settings.
