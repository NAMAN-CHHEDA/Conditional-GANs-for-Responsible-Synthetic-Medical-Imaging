"""Matplotlib figures shared by the scripts."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, RED, GREY, GREEN, ORANGE = "#4C78A8", "#E45756", "#9D9D9D", "#54A24B", "#F58518"


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def plot_training_history(history: list[dict], path: str, title: str = "cGAN training") -> None:
    ep = [h["epoch"] for h in history]
    get = lambda k: [h.get(k, np.nan) for h in history]  # noqa: E731
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    ax = axes[0, 0]
    ax.plot(ep, get("d_loss"), color=BLUE, label="D loss")
    ax.plot(ep, get("g_loss"), color=RED, label="G loss")
    ax.set_title("adversarial losses")
    ax.legend(frameon=False)
    ax = axes[0, 1]
    ax.plot(ep, get("d_acc_real"), color=BLUE, label="D acc real")
    ax.plot(ep, get("d_acc_fake"), color=RED, label="D acc fake")
    ax.plot(ep, get("d_real"), color=BLUE, ls="--", label="mean D(x)")
    ax.plot(ep, get("d_fake"), color=RED, ls="--", label="mean D(G(z))")
    ax.axhline(0.5, color=GREY, lw=0.8)
    ax.set_ylim(0, 1.02)
    ax.set_title("discriminator balance")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 2]
    ax.plot(ep, get("g_grad_norm"), color=ORANGE)
    ax.set_title("generator gradient norm")
    ax = axes[1, 0]
    ax.plot(ep, get("fd_0"), "o-", ms=3, color=BLUE, label="normal")
    ax.plot(ep, get("fd_1"), "o-", ms=3, color=RED, label="pneumonia")
    ax.set_title("Frechet distance (judge features, lower=better)")
    ax.legend(frameon=False)
    ax = axes[1, 1]
    ax.plot(ep, get("div_ratio_0"), "o-", ms=3, color=BLUE, label="diversity ratio normal")
    ax.plot(ep, get("div_ratio_1"), "o-", ms=3, color=RED, label="diversity ratio pneumonia")
    ax.plot(ep, get("class_consistency"), "o-", ms=3, color=GREEN, label="class consistency")
    ax.axhline(1.0, color=GREY, lw=0.8)
    ax.set_title("diversity / conditioning")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 2]
    ax.plot(ep, get("mem_rate"), "o-", ms=3, color=ORANGE, label="memorisation flag rate")
    ax.plot(ep, get("val_f1_gain"), "o-", ms=3, color=GREEN, label="val F1 gain")
    ax.plot(ep, get("quality_score"), "o-", ms=3, color="black", label="quality score")
    ax.axhline(0.05, color=GREY, lw=0.8, ls=":")
    ax.axhline(0.0, color=GREY, lw=0.8)
    ax.set_title("selection signals")
    ax.legend(frameon=False, fontsize=8)
    for a in axes.flat:
        _clean(a)
        a.set_xlabel("epoch")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
