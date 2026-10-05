from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

from label_noise_shape import (
    generate_semantic2_noise,
    generate_symmetric_noise,
    generate_xia_idln_noise,
)

PAPER_BUDGETS = {
    "aggregate": 4_505,
    "random2": 9_061,
    "worst": 20_104,
}


def save_result(
    path: Path,
    result,
    *,
    method: str,
    budget_name: str,
    seed: int,
) -> dict[str, object]:
    payload = {
        "labels_clean": result.clean_labels.numpy(),
        "labels_noisy": result.noisy_labels.numpy(),
        "flipped_mask": result.flipped_mask.numpy(),
        "selected_indices": result.selected_indices.numpy(),
        "empirical_transition": result.empirical_transition.numpy(),
    }
    if result.intended_transition is not None:
        payload["intended_transition"] = result.intended_transition.numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {
        "method": method,
        "budget_name": budget_name,
        "budget": result.flip_count,
        "noise_rate": result.flip_count / result.clean_labels.numel(),
        "seed": seed,
        "output": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the paper's synthetic CIFAR-10 baseline labels"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/baseline_labels")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("symmetric", "semantic2", "idln"),
        default=("symmetric", "semantic2", "idln"),
    )
    args = parser.parse_args()

    dataset = datasets.CIFAR10(root=args.data_root, train=True, download=True)
    labels = torch.as_tensor(dataset.targets, dtype=torch.long)
    raw_inputs = torch.from_numpy(dataset.data)

    summaries = []
    for seed in args.seeds:
        for budget_name, budget in PAPER_BUDGETS.items():
            for method in args.methods:
                if method == "symmetric":
                    result = generate_symmetric_noise(
                        labels, budget=budget, num_classes=10, seed=seed
                    )
                elif method == "semantic2":
                    result = generate_semantic2_noise(labels, budget=budget, seed=seed)
                else:
                    result = generate_xia_idln_noise(
                        raw_inputs,
                        labels,
                        budget=budget,
                        num_classes=10,
                        seed=seed,
                    )
                output = args.output_dir / f"{method}_{budget_name}_seed{seed}.npz"
                summaries.append(
                    save_result(
                        output,
                        result,
                        method=method,
                        budget_name=budget_name,
                        seed=seed,
                    )
                )
                print(f"saved {output} ({result.flip_count} flips)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2)


if __name__ == "__main__":
    main()
