from __future__ import annotations

from pathlib import Path

from label_noise_shape.reference import load_accuracy_rows, validate_accuracy_rows


def test_reference_table_contract() -> None:
    rows = load_accuracy_rows(Path("results/reference/table1_accuracy.csv"))
    validate_accuracy_rows(rows)
