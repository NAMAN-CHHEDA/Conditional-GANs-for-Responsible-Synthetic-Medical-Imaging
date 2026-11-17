"""Downstream classifier training / evaluation (the task-grounded test of the GAN)."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)

from .models import SmallCNN


def random_augment(x01: torch.Tensor) -> torch.Tensor:
    """'Naive' augmentation: small rotation/translation/scale + brightness/contrast jitter.
    Horizontal flips are deliberately NOT used: they move the heart to the wrong side and
    create anatomically implausible chest X-rays."""
    b = x01.shape[0]
    dev = x01.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(10)
    scale = 1 + (torch.rand(b, device=dev) * 2 - 1) * 0.05
    tx = (torch.rand(b, device=dev) * 2 - 1) * 0.08
    ty = (torch.rand(b, device=dev) * 2 - 1) * 0.08
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x01.shape), align_corners=False)
    x = F.grid_sample(x01, grid, padding_mode="border", align_corners=False)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    c = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * 0.1
    br = (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * 0.05
    return ((x - mean) * c + mean + br).clamp(0, 1)


@torch.no_grad()
def predict_proba(model: torch.nn.Module, x01: torch.Tensor, device, batch_size: int = 2048) -> np.ndarray:
    """Returns P(pneumonia) = P(class 1)."""
    model.eval()
    out = []
    for i in range(0, len(x01), batch_size):
        logits = model(x01[i:i + batch_size].to(device))
        out.append(torch.softmax(logits.float(), 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def compute_metrics(y_true, prob1, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob1 = np.asarray(prob1, dtype=float)
    pred = (prob1 >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, prob1)) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "auc": auc,
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_normal": float(f1_score(y_true, pred, pos_label=0, zero_division=0)),
        "f1_pneumonia": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def train_classifier(x: torch.Tensor, y: torch.Tensor, device, epochs: int = 15, seed: int = 0,
                     batch_size: int = 128, lr: float = 1e-3, weight_decay: float = 1e-4,
                     class_weight=None, augment: bool = False,
                     x_val: torch.Tensor | None = None, y_val: torch.Tensor | None = None,
                     select_metric: str = "f1_macro"):
    """Train SmallCNN. If validation data is given, keep the epoch with best `select_metric`
    (model selection uses validation only; the test set is never touched here)."""
    torch.manual_seed(seed)
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x, y = x.to(device), y.to(device)
    w = None if class_weight is None else torch.as_tensor(class_weight, dtype=torch.float32, device=device)
    g = torch.Generator().manual_seed(seed)
    best_score, best_state, best_epoch, history = -1.0, None, -1, []
    n = len(y)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            xb = x[idx]
            if augment:
                xb = random_augment(xb)
            loss = F.cross_entropy(model(xb), y[idx], weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if x_val is not None:
            m = compute_metrics(y_val.cpu().numpy(), predict_proba(model, x_val, device))
            history.append({"epoch": ep + 1, **{k: v for k, v in m.items() if k != "confusion_matrix"}})
            if m[select_metric] > best_score:
                best_score, best_epoch = m[select_metric], ep + 1
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val": best_score, "history": history}


def inverse_frequency_weights(y: torch.Tensor, n_classes: int = 2) -> list[float]:
    counts = torch.bincount(y.cpu(), minlength=n_classes).float()
    return (counts.sum() / (n_classes * counts.clamp(min=1))).tolist()
