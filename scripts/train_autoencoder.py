from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from label_noise_shape.autoencoder import CifarResNetAutoencoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the paper's CIFAR-10 autoencoder"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/cifar10_resnet18_ae_128d.pt"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CifarResNetAutoencoder(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    losses: list[float] = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        sample_count = 0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False)
        for images, _ in progress:
            images = images.to(device, non_blocking=True)
            reconstruction = model(images)
            loss = criterion(reconstruction, images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.shape[0]
            sample_count += images.shape[0]
            progress.set_postfix(mse=f"{loss.item():.5f}")
        mean_loss = total_loss / max(sample_count, 1)
        losses.append(mean_loss)
        print(f"epoch={epoch + 1:03d} mse={mean_loss:.7f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "latent_dim": args.latent_dim,
            "hard_normalize_latent": False,
            "sphere_lambda": 0.0,
            "architecture": "cifar_resnet18",
            "dataset": "cifar10",
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "losses": losses,
        },
        args.output,
    )
    print(f"saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
