"""Framework-agnostic core of the synthetic-image service (validated request model,
batched generation, PNG encoding with provenance metadata). Ray Serve wraps this in
serve_app.py; unit tests exercise it without Ray."""
from __future__ import annotations

import base64
import io
import secrets
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, PngImagePlugin
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import LABEL_NAMES, NAME_TO_LABEL, SYNTHETIC_DISCLAIMER, __version__
from .models import load_generator, sample_latent
from .utils import file_sha256

MAX_IMAGES_PER_REQUEST = 64


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Union[int, str] = Field(..., description="0/'normal' or 1/'pneumonia'")
    n: int = Field(8, ge=1, le=MAX_IMAGES_PER_REQUEST, description="number of images")
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1, description="optional RNG seed for reproducibility")
    size: int = Field(64, description="output resolution: 28 (dataset native) or 64 (generator native)")
    return_grid: bool = Field(False, description="also return one PNG grid of all images")

    @field_validator("label", mode="before")
    @classmethod
    def _label(cls, v):
        if isinstance(v, bool):
            raise ValueError("label must be 0, 1, 'normal' or 'pneumonia'")
        if isinstance(v, str):
            key = v.strip().lower()
            if key.isdigit():
                v = int(key)
            elif key in NAME_TO_LABEL:
                return NAME_TO_LABEL[key]
            else:
                raise ValueError("label must be 0, 1, 'normal' or 'pneumonia'")
        if not isinstance(v, int) or v not in LABEL_NAMES:
            raise ValueError("label must be 0 (normal) or 1 (pneumonia)")
        return int(v)

    @field_validator("size")
    @classmethod
    def _size(cls, v):
        if v not in (28, 64):
            raise ValueError("size must be 28 or 64")
        return v


def encode_png(img_u8: np.ndarray, label: int, seed: int, index: int, model_id: str) -> str:
    """PNG with tEXt provenance chunks so the 'synthetic' flag travels with the file."""
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Synthetic", "true")
    meta.add_text("Disclaimer", SYNTHETIC_DISCLAIMER)
    meta.add_text("Generator", f"cDCGAN-PneumoniaMNIST/{model_id}")
    meta.add_text("Label", LABEL_NAMES[label])
    meta.add_text("Seed", f"{seed}:{index}")
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img_u8, dtype=np.uint8)).save(buf, format="PNG", pnginfo=meta)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_grid_u8(imgs: np.ndarray, nrow: int = 8, pad: int = 2) -> np.ndarray:
    n, h, w = imgs.shape
    ncol = min(nrow, n)
    nr = int(np.ceil(n / ncol))
    grid = np.full((nr * (h + pad) + pad, ncol * (w + pad) + pad), 255, dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, ncol)
        grid[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = imgs[i]
    return grid


class SyntheticImageService:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.G, self.cfg, self.meta = load_generator(checkpoint_path, self.device)
        self.model_id = file_sha256(checkpoint_path)[:12]
        self.loaded_at = time.time()
        with torch.no_grad():  # warm-up (CUDA context, cudnn autotune)
            self.G(torch.zeros(2, self.G.latent_dim, device=self.device),
                   torch.tensor([0, 1], device=self.device))

    @torch.no_grad()
    def generate_batch(self, reqs: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """reqs: list of (label, n, seed, size). One forward pass for the whole batch.
        The generator is in eval mode (BatchNorm uses running statistics), so each
        request's images depend only on its own (label, seed): batching with other
        requests does not change outputs."""
        zs, ys = [], []
        for label, n, seed, _ in reqs:
            zs.append(sample_latent(n, self.G.latent_dim, seed, "cpu"))
            ys.append(torch.full((n,), label, dtype=torch.long))
        z = torch.cat(zs).to(self.device)
        y = torch.cat(ys).to(self.device)
        x = self.G(z, y).float()
        out, i = [], 0
        for label, n, seed, size in reqs:
            xi = x[i:i + n]
            i += n
            if size != xi.shape[-1]:
                xi = torch.nn.functional.interpolate((xi + 1) / 2, size=(size, size), mode="area") * 2 - 1
            out.append(((xi.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)[:, 0].cpu().numpy())
        return out

    def build_response(self, req: GenerateRequest, seed: int, imgs: np.ndarray, t_start: float,
                       replica: str | None = None) -> dict:
        resp = {
            "synthetic": True,
            "disclaimer": SYNTHETIC_DISCLAIMER,
            "label": req.label,
            "label_name": LABEL_NAMES[req.label],
            "n": req.n,
            "seed": seed,
            "image_size": req.size,
            "format": "png_base64",
            "images": [encode_png(im, req.label, seed, k, self.model_id) for k, im in enumerate(imgs)],
            "model": {"name": "conditional-DCGAN (projection D)", "id": self.model_id,
                      "package_version": __version__,
                      "selected_epoch": self.meta.get("selected_epoch", self.meta.get("epoch"))},
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
        }
        if replica:
            resp["replica"] = replica
        if req.return_grid:
            resp["grid"] = encode_png(make_grid_u8(imgs), req.label, seed, -1, self.model_id)
        return resp

    def handle(self, req: GenerateRequest) -> dict:
        """Synchronous single-request path (used by tests and non-Ray usage)."""
        t0 = time.perf_counter()
        seed = req.seed if req.seed is not None else secrets.randbelow(2**31 - 1)
        imgs = self.generate_batch([(req.label, req.n, seed, req.size)])[0]
        return self.build_response(req, seed, imgs, t0)

    def model_card(self) -> dict:
        return {
            "name": "Synthetic PneumoniaMNIST generator",
            "intended_use": "Education, teaching demos, and research prototyping (e.g. stress-testing pipelines).",
            "not_intended_for": ["diagnosis or triage", "clinical decision support", "training models deployed "
                                 "on patients without real-data validation", "presenting outputs as real patient images"],
            "training_data": "PneumoniaMNIST train split (MedMNIST v2; pediatric CXR, 28x28, upsampled to "
                             f"{self.cfg['img_size']}x{self.cfg['img_size']})",
            "labels": LABEL_NAMES,
            "config": self.cfg,
            "selection": {k: v for k, v in self.meta.items() if k != "metrics_at_selection"},
            "disclaimer": SYNTHETIC_DISCLAIMER,
            "max_images_per_request": MAX_IMAGES_PER_REQUEST,
        }
