from __future__ import annotations

import csv
from pathlib import Path

LEARNERS = ("ce", "gce", "coteaching", "snv_f", "promix")
BUDGETS = ("aggregate", "random2", "worst")


def load_accuracy_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_accuracy_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 19:
        raise ValueError(f"expected 19 table rows, found {len(rows)}")
    clean = [row for row in rows if row["budget"] == "clean"]
    if len(clean) != 1 or clean[0]["noise"] != "Clean dataset":
        raise ValueError("reference table must contain exactly one clean row")

    for budget in BUDGETS:
        block = [row for row in rows if row["budget"] == budget]
        if len(block) != 6:
            raise ValueError(f"{budget} must contain six noise variants")
        margin = next(row for row in block if row["noise"] == "MarginNoise")
        for learner in LEARNERS:
            values = [float(row[f"{learner}_mean"]) for row in block]
            if float(margin[f"{learner}_mean"]) != min(values):
                raise ValueError(
                    f"MarginNoise is not the minimum for {budget}/{learner}"
                )
