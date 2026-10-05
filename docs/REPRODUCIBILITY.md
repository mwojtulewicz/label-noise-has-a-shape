# Reproducibility guide

## Exact method definition

The public API mirrors the camera-ready paper:

1. Extract one 128-dimensional latent vector for every CIFAR-10 training image
   with the frozen autoencoder encoder.
2. Subtract the coordinate-wise mean over all 50,000 training embeddings.
3. L2-normalize every centered embedding.
4. Average normalized embeddings within each clean class and L2-normalize the
   ten resulting prototypes.
5. Use `beta = 2.5` and one of the exact budgets 4,505, 9,061, or 20,104.

ProtoNoise samples the budgeted examples uniformly without replacement. Its
wrong-label distribution is a softmax over class-prototype cosine similarities.
MarginNoise selects the examples with the smallest signed margin
`s(true) - max(s(wrong))`; its destination distribution is a softmax over each
selected example's wrong-class prototype similarities.

## Seeds and exact budgets

Generator seeds are explicit and affect uniform example sampling or label
destination draws. MarginNoise selection itself is deterministic for a fixed
embedding matrix, except for the ordering of exact floating-point ties.

The paper matches the integer numbers of changed CIFAR-10N labels rather than
rounded nominal rates:

| Human reference | Flip count | Rate over 50,000 |
|---|---:|---:|
| Aggregate | 4,505 | 9.010% |
| Random 2 | 9,061 | 18.122% |
| Worst | 20,104 | 40.208% |

The synthetic comparison baselines use the same integer budgets. Symmetric and
Semantic-2 generation is deterministic for a fixed seed. IdLN also depends on
the raw CIFAR-10 array layout and scaling: this release uses torchvision's
`uint8` NHWC array, flattened per image and divided by 255, matching the paper
pipeline.

IdLN is an exact-budget adaptation of Algorithm 2 in Xia et al., *Part-dependent
Label Noise: Towards Instance-dependent Label Noise* (NeurIPS 2020). The
published per-example propensities are retained, but they weight sampling
without replacement instead of independent Bernoulli decisions. This change is
what makes the comparison use exactly the same number of flips.

## Representation reproducibility

The paper checkpoint has SHA-256 digest
`8bf4568574185265482c9dd03e37b84350298e8bcdc4bf405c7815cb31cc79a5`.
The original environment used Python 3.12, PyTorch 2.10.0 with CUDA 12.8, and
torchvision 0.25.0. Exact labels require the same checkpoint; retraining can
change the learned geometry even when the training seed and recipe match.

The 52 MiB checkpoint is intentionally excluded from normal Git history. It
should be distributed as a checksum-verified GitHub release asset.

## What is not stored

CIFAR-10 images, CIFAR-10N annotations, third-party robust-learning
repositories, checkpoints, embeddings, and raw training logs are not committed.
Scripts download CIFAR-10 through torchvision. CIFAR-10N and third-party learner
implementations retain their original licenses and should be obtained from
their official repositories.
