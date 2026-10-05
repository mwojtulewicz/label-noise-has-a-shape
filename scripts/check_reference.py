from __future__ import annotations

from pathlib import Path

from label_noise_shape.reference import load_accuracy_rows, validate_accuracy_rows

REFERENCE = Path("results/reference/table1_accuracy.csv")


def main() -> None:
    rows = load_accuracy_rows(REFERENCE)
    validate_accuracy_rows(rows)
    print("reference table validated: 19 rows and all headline minima")


if __name__ == "__main__":
    main()
