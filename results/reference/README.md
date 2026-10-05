# Reference artifacts

This directory contains compact, publication-facing summaries that can be
audited without distributing raw datasets, model checkpoints, or complete
experiment directories.

- `table1_accuracy.csv` is the camera-ready clean-test accuracy table in a
  machine-readable wide format. CE, GCE, Co-teaching, and SNV-F use three seeds;
  ProMix uses two seeds. Values are percentages reported as mean and sample
  standard deviation.

Run `uv run python scripts/check_reference.py` to validate its coverage and the
headline per-budget MarginNoise minimum.
