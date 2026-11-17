"""Conditional DCGAN generator / projection discriminator, plus a small CNN classifier.

Design choices (justified in the report):
* Generator G(z, y): the class label is embedded (nn.Embedding) and concatenated with z,
  then upsampled with the DCGAN stack ConvT-BN-ReLU and a Tanh output (pairs with the
  [-1,1] input normalisation).
* Discriminator D(x, y): DCGAN conv stack + *projection* conditioning (Miyato & Koyama,
  2018): logit = w^T h + <embed(y), h>. Projection gives D a class-specific direction in
  feature space instead of just concatenating a label map, and is the standard partner
  for spectral normalisation. Spectral norm is optional (searched by Ray Tune); when it
  is on, BatchNorm is removed from D.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


def _sn(module: nn.Module, enabled: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if enabled else module


def _weight(m: nn.Module) -> torch.Tensor:
    return m.weight_orig if hasattr(m, "weight_orig") else m.weight


def dcgan_init(m: nn.Module, init_embedding: bool = False) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(_weight(m), 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding) and init_embedding:
        nn.init.normal_(_weight(m), 0.0, 0.02)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 64, n_classes: int = 2, ngf: int = 64,
                 img_size: int = 64, embed_dim: int = 32):
        super().__init__()
        if img_size not in (32, 64):
            raise ValueError("img_size must be 32 or 64")
        self.latent_dim, self.n_classes, self.img_size = latent_dim, n_classes, img_size
        self.embed = nn.Embedding(n_classes, embed_dim)
        n_up = int(math.log2(img_size)) - 2
        ch = ngf * 2 ** (n_up - 1)
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(latent_dim + embed_dim, ch, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(True),
        ]
        for _ in range(n_up - 1):
            layers += [nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        layers += [nn.ConvTranspose2d(ch, 1, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.apply(lambda m: dcgan_init(m, init_embedding=False))

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, self.embed(y)], dim=1)[:, :, None, None]
        return self.net(h)


class Discriminator(nn.Module):
    def __init__(self, n_classes: int = 2, ndf: int = 64, img_size: int = 64,
                 spectral_norm: bool = False):
        super().__init__()
        sn = bool(spectral_norm)
        n_down = int(math.log2(img_size)) - 2
        layers: list[nn.Module] = [_sn(nn.Conv2d(1, ndf, 4, 2, 1), sn), nn.LeakyReLU(0.2, True)]
        ch = ndf
        for _ in range(n_down - 1):
            layers.append(_sn(nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=sn), sn))
            if not sn:
                layers.append(nn.BatchNorm2d(ch * 2))
            layers.append(nn.LeakyReLU(0.2, True))
            ch *= 2
        self.features = nn.Sequential(*layers)
        self.linear = _sn(nn.Linear(ch, 1), sn)
        self.embed = _sn(nn.Embedding(n_classes, ch), sn)
        self.apply(lambda m: dcgan_init(m, init_embedding=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.features(x).sum(dim=(2, 3))  # global sum pooling -> (B, ch)
        return self.linear(h).squeeze(1) + (self.embed(y) * h).sum(dim=1)


class SmallCNN(nn.Module):
    """Small 28x28 classifier used (a) as the downstream model and (b) as the frozen
    'judge' feature extractor for Frechet distance / diversity / class consistency."""

    def __init__(self, n_classes: int = 2, width: int = 32, feat_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        w = width
        self.body = nn.Sequential(
            nn.Conv2d(1, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(True),
            nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * w, 4 * w, 3, padding=1), nn.BatchNorm2d(4 * w), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.Linear(4 * w, feat_dim), nn.ReLU(True), nn.Dropout(dropout))
        self.head = nn.Linear(feat_dim, n_classes)

    def features(self, x01: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(x01 * 2.0 - 1.0))

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x01))


# ----------------------------------------------------------------------------- helpers

def build_gan(cfg: dict) -> tuple[Generator, Discriminator]:
    G = Generator(latent_dim=cfg["latent_dim"], ngf=cfg["ngf"], img_size=cfg["img_size"],
                  embed_dim=cfg.get("embed_dim", 32))
    D = Discriminator(ndf=cfg["ndf"], img_size=cfg["img_size"], spectral_norm=cfg["spectral_norm"])
    return G, D


def sample_latent(n: int, latent_dim: int, seed: int, device="cpu") -> torch.Tensor:
    """Latents drawn from a seeded *CPU* generator so a seed gives the same z on any device."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(n, latent_dim, generator=g).to(device)


@torch.no_grad()
def generate(G: Generator, labels: Iterable[int] | torch.Tensor, seed: int, device=None,
             batch_size: int = 512) -> torch.Tensor:
    """Generate images in GAN space ([-1,1]) for the given labels; returns a CPU tensor."""
    device = device or next(G.parameters()).device
    labels = torch.as_tensor(list(labels) if not torch.is_tensor(labels) else labels, dtype=torch.long)
    z = sample_latent(len(labels), G.latent_dim, seed, "cpu")
    out = []
    for i in range(0, len(labels), batch_size):
        out.append(G(z[i:i + batch_size].to(device), labels[i:i + batch_size].to(device)).cpu())
    return torch.cat(out) if out else torch.empty(0, 1, G.img_size, G.img_size)


def save_generator(path: str, G: Generator, cfg: dict, meta: dict | None = None) -> None:
    torch.save({"config": cfg, "G": G.state_dict(), "meta": meta or {}}, path)


def load_generator(path: str, device="cpu") -> tuple[Generator, dict, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    G = Generator(latent_dim=cfg["latent_dim"], ngf=cfg["ngf"], img_size=cfg["img_size"],
                  embed_dim=cfg.get("embed_dim", 32))
    G.load_state_dict(state["G"])
    return G.to(device).eval(), cfg, state.get("meta", {})


def load_judge(path_or_state, device="cpu") -> SmallCNN:
    """Load the frozen real-data judge from a path or from an already-loaded state dict."""
    if isinstance(path_or_state, str):
        path_or_state = torch.load(path_or_state, map_location="cpu", weights_only=False)["state_dict"]
    m = SmallCNN()
    m.load_state_dict(path_or_state)
    return m.to(device).eval()


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
