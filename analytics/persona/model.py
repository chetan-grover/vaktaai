"""E5: small VAE over standardised 24-d session vectors (latent 2-4)."""
from __future__ import annotations

import torch
from torch import nn


class SessionVAE(nn.Module):
    def __init__(self, in_dim: int = 24, hidden: int = 64, latent: int = 2) -> None:
        super().__init__()
        self.config = {"in_dim": in_dim, "hidden": hidden, "latent": latent}
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden // 2), nn.GELU())
        self.mu = nn.Linear(hidden // 2, latent)
        self.logvar = nn.Linear(hidden // 2, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hidden),
                                     nn.GELU(), nn.Linear(hidden, in_dim))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(-8, 8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decoder(z), mu, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 0.5) -> torch.Tensor:
    rec = ((recon - x) ** 2).sum(dim=1).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return rec + beta * kl
