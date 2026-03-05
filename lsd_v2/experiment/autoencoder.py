from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import AutoencoderConfig


class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


@dataclass
class TrainedAutoencoder:
    model: MLPAutoencoder
    mean: np.ndarray
    std: np.ndarray

    def encode(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        x_norm = (x - self.mean) / self.std
        dataset = TensorDataset(torch.from_numpy(x_norm.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        latents = []
        with torch.no_grad():
            for (batch,) in loader:
                _, z = self.model(batch)
                latents.append(z.cpu().numpy())
        return np.concatenate(latents, axis=0)


def train_autoencoder(x_train: np.ndarray, cfg: AutoencoderConfig) -> TrainedAutoencoder:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_norm = (x_train - mean) / std

    input_dim = x_norm.shape[1]
    hidden_dim = max(cfg.latent_dim * cfg.hidden_multiplier, cfg.latent_dim + 32)
    model = MLPAutoencoder(input_dim=input_dim, latent_dim=cfg.latent_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(x_norm.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(cfg.epochs):
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()

    return TrainedAutoencoder(model=model, mean=mean, std=std)

