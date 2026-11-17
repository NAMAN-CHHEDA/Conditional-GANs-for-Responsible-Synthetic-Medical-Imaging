"""PneumoniaMNIST loading and preprocessing.

Everything is held as in-memory tensors (the dataset is tiny: 4708/524/624 images),
which makes GAN training fast and lets Ray Tune ship the data to trials through the
object store instead of every trial re-reading files.

Spaces used in this project
---------------------------
* ``x28``   : original 28x28 images scaled to [0, 1].
* GAN space : images resized to ``img_size`` (default 64, as in the assignment) and
              normalised to [-1, 1] so they match the generator's Tanh output.
* ``rt28``  : "round-trip" images (28 -> img_size -> 28). Synthetic images are produced
              at img_size and downsampled to 28 for the classifier. Real images used by
              classifiers / memorisation checks go through the same round-trip so that a
              downstream classifier cannot use "slightly blurrier = synthetic = normal"
              as a shortcut.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

SPLITS = ("train", "val", "test")


@dataclass
class SplitData:
    x28: torch.Tensor  # float32 (N,1,28,28) in [0,1]
    y: torch.Tensor  # int64 (N,)  0=normal, 1=pneumonia

    def __len__(self) -> int:
        return int(self.y.shape[0])


def get_label_mapping() -> dict[int, str]:
    """Label mapping straight from the MedMNIST metadata (do not hard-code blindly)."""
    from medmnist import INFO

    return {int(k): v for k, v in INFO["pneumoniamnist"]["label"].items()}


def load_pneumoniamnist(root: str = "data", download: bool = True) -> dict[str, SplitData]:
    from medmnist import PneumoniaMNIST

    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    out: dict[str, SplitData] = {}
    for split in SPLITS:
        ds = PneumoniaMNIST(split=split, download=download, root=root)
        imgs = np.asarray(ds.imgs)
        if imgs.ndim == 4:  # (N,H,W,1) in some versions
            imgs = imgs[..., 0]
        x = torch.from_numpy(imgs).float().div(255.0).unsqueeze(1).contiguous()
        y = torch.from_numpy(np.asarray(ds.labels).reshape(-1)).long()
        out[split] = SplitData(x, y)
    return out


def to_gan_space(x01: torch.Tensor, img_size: int = 64) -> torch.Tensor:
    """[0,1] 28x28 -> [-1,1] img_size x img_size (bilinear upsampling)."""
    x = x01
    if x.shape[-1] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x * 2.0 - 1.0


def from_gan_space(x: torch.Tensor, out_size: int = 28) -> torch.Tensor:
    """[-1,1] GAN space -> [0,1] at out_size (area downsampling = anti-aliased)."""
    x01 = (x.clamp(-1, 1) + 1.0) / 2.0
    if x01.shape[-1] != out_size:
        x01 = F.interpolate(x01, size=(out_size, out_size), mode="area")
    return x01


def roundtrip(x01: torch.Tensor, img_size: int = 64) -> torch.Tensor:
    if img_size == x01.shape[-1]:
        return x01.clone()
    return from_gan_space(to_gan_space(x01, img_size), x01.shape[-1])


def class_counts(y: torch.Tensor, n_classes: int = 2) -> list[int]:
    return [int((y == c).sum()) for c in range(n_classes)]


def dataset_summary(splits: dict[str, SplitData]) -> dict:
    mapping = {0: "normal", 1: "pneumonia"}
    try:
        mapping = get_label_mapping()
    except Exception:  # pragma: no cover - medmnist missing in some unit tests
        pass
    summary = {"label_mapping": mapping, "splits": {}}
    for name, s in splits.items():
        counts = class_counts(s.y)
        summary["splits"][name] = {
            "n": len(s),
            "counts": {mapping[c]: counts[c] for c in range(2)},
            "fraction_normal": counts[0] / max(1, len(s)),
            "imbalance_ratio_pneumonia_to_normal": counts[1] / max(1, counts[0]),
            "shape": list(s.x28.shape[1:]),
            "pixel_min": float(s.x28.min()),
            "pixel_max": float(s.x28.max()),
            "pixel_mean": float(s.x28.mean()),
            "pixel_std": float(s.x28.std()),
        }
    return summary
