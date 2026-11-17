"""Small shared helpers: seeding, devices, JSON IO, image grids."""
from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


def save_json(obj: Any, path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensor (N,1,H,W) -> uint8 numpy (N,H,W)."""
    x = ((x.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x[:, 0].numpy()


def save_grid(x: torch.Tensor, path: str, nrow: int = 8, value_range=(-1, 1)) -> None:
    """Save a tensor batch as an image grid (expects GAN space [-1,1] by default)."""
    from torchvision.utils import save_image

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    lo, hi = value_range
    save_image((x.detach().float().cpu() - lo) / (hi - lo), path, nrow=nrow, padding=2, pad_value=1.0)


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
