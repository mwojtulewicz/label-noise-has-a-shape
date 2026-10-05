from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from label_noise_shape import (
    fit_geometry,
    generate_margin_noise,
    generate_proto_noise,
)
from label_noise_shape.autoencoder import CifarResNetAutoencoder

PAPER_CHECKPOINT_SHA256 = (
    "8bf4568574185265482c9dd03e37b84350298e8bcdc4bf405c7815cb31cc79a5"
)
PAPER_BUDGETS = {
    "aggregate": 4_505,
    "random2": 9_061,
    "worst": 20_104,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_encoder(checkpoint: Path, device: torch.device) -> CifarResNetAutoencoder:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("architecture") != "cifar_resnet18":
        raise ValueError("checkpoint is not a cifar_resnet18 autoencoder")
    latent_dim = int(state.get("latent_dim", 128))
    model = CifarResNetAutoencoder(latent_dim=latent_dim)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def extract_embeddings(
    model: CifarResNetAutoencoder,
    dataset: datasets.CIFAR10,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="extract embeddings"):
            embeddings.append(model.encoder(images.to(device, non_blocking=True)).cpu())
            labels.append(targets.cpu())
    return torch.cat(embeddings), torch.cat(labels)


def save_result(
    path: Path,
    result,
    geometry,
    *,
    method: str,
    budget_name: str,
    beta: float,
    seed: int,
    checkpoint_digest: str,
) -> dict[str, object]:
    payload = {
        "labels_clean": result.clean_labels.numpy(),
        "labels_noisy": result.noisy_labels.numpy(),
        "flipped_mask": result.flipped_mask.numpy(),
        "selected_indices": result.selected_indices.numpy(),
        "empirical_transition": result.empirical_transition.numpy(),
        "prototypes": geometry.prototypes.numpy(),
        "embedding_mean": geometry.embedding_mean.numpy(),
    }
    if result.intended_transition is not None:
        payload["intended_transition"] = result.intended_transition.numpy()
    if result.margins is not None:
        payload["margins"] = result.margins.numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {
        "method": method,
        "budget_name": budget_name,
        "budget": result.flip_count,
        "noise_rate": result.flip_count / result.clean_labels.numel(),
        "beta": beta,
        "seed": seed,
        "checkpoint_sha256": checkpoint_digest,
        "output": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the paper's CIFAR-10 labels")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_labels"))
    parser.add_argument("--beta", type=float, default=2.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--require-paper-checkpoint",
        action="store_true",
        help="fail unless the checkpoint exactly matches the paper artifact",
    )
    args = parser.parse_args()

    digest = sha256(args.checkpoint)
    if args.require_paper_checkpoint and digest != PAPER_CHECKPOINT_SHA256:
        raise SystemExit(
            "checkpoint SHA-256 does not match the paper artifact:\n"
            f"expected {PAPER_CHECKPOINT_SHA256}\nfound    {digest}"
        )
    print(f"checkpoint SHA-256: {digest}")

    device = resolve_device(args.device)
    dataset = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    model = load_encoder(args.checkpoint, device)
    embeddings, labels = extract_embeddings(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    geometry = fit_geometry(embeddings, labels, num_classes=10)

    summaries = []
    for seed in args.seeds:
        for budget_name, budget in PAPER_BUDGETS.items():
            proto = generate_proto_noise(
                labels, geometry, budget=budget, beta=args.beta, seed=seed
            )
            margin = generate_margin_noise(
                labels, geometry, budget=budget, beta=args.beta, seed=seed
            )
            for method, result in (("protonoise", proto), ("marginnoise", margin)):
                output = args.output_dir / f"{method}_{budget_name}_seed{seed}.npz"
                summaries.append(
                    save_result(
                        output,
                        result,
                        geometry,
                        method=method,
                        budget_name=budget_name,
                        beta=args.beta,
                        seed=seed,
                        checkpoint_digest=digest,
                    )
                )
                print(f"saved {output} ({result.flip_count} flips)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2)


if __name__ == "__main__":
    main()
