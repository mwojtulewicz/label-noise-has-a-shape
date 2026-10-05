from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class CifarResNetEncoder(nn.Module):
    """CIFAR-stem ResNet-18 followed by an unrestricted signed bottleneck."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.projection = nn.Linear(model.fc.in_features, latent_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = torch.flatten(self.backbone(inputs), 1)
        return self.projection(features)


class CifarResNetDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.fc(latent).view(latent.size(0), 256, 4, 4)
        return self.deconv(features)


class CifarResNetAutoencoder(nn.Module):
    """ResNet-18-like CIFAR autoencoder used by the paper."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = CifarResNetEncoder(latent_dim=latent_dim)
        self.decoder = CifarResNetDecoder(latent_dim=latent_dim)

    def forward(
        self, inputs: torch.Tensor, *, return_latent: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(inputs)
        reconstruction = self.decoder(latent)
        if return_latent:
            return reconstruction, latent
        return reconstruction
