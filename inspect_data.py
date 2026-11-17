"""Step 1 - load and inspect PneumoniaMNIST.

python inspect_data.py --out outputs/data
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gan_lab.data import SPLITS, class_counts, dataset_summary, load_pneumoniamnist, to_gan_space
from gan_lab.utils import save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="outputs/data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    splits = load_pneumoniamnist(args.data_root)
    summary = dataset_summary(splits)

    # The assignment's torchvision pipeline: confirm shape and [-1,1] normalisation.
    from medmnist import PneumoniaMNIST
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    train_ds = PneumoniaMNIST(split="train", transform=transform, download=True,
                              root=os.path.abspath(args.data_root))
    x0, y0 = train_ds[0]
    xs = torch.stack([train_ds[i][0] for i in range(256)])
    ours = to_gan_space(splits["train"].x28[:256], 64)
    summary["torchvision_check"] = {
        "len_train": len(train_ds), "sample_shape": list(x0.shape), "sample_label": int(np.asarray(y0).item()),
        "min": float(xs.min()), "max": float(xs.max()),
        "mean_abs_diff_vs_tensor_pipeline": float((xs - ours).abs().mean()),
    }
    save_json(summary, os.path.join(args.out, "data_summary.json"))

    print(f"Label mapping: {summary['label_mapping']}")
    for s in SPLITS:
        info = summary["splits"][s]
        print(f"{s:5s}: n={info['n']:5d} counts={info['counts']} "
              f"normal fraction={info['fraction_normal']:.3f} shape={info['shape']}")
    print("torchvision check:", summary["torchvision_check"])

    # ---- class counts
    fig, ax = plt.subplots(figsize=(6, 3.5))
    w = 0.38
    xs_ = np.arange(len(SPLITS))
    c0 = [class_counts(splits[s].y)[0] for s in SPLITS]
    c1 = [class_counts(splits[s].y)[1] for s in SPLITS]
    b0 = ax.bar(xs_ - w / 2, c0, w, label="0 = normal", color="#4C78A8")
    b1 = ax.bar(xs_ + w / 2, c1, w, label="1 = pneumonia", color="#E45756")
    ax.bar_label(b0, fontsize=8)
    ax.bar_label(b1, fontsize=8)
    ax.set_xticks(xs_, SPLITS)
    ax.set_ylabel("images")
    ax.set_title("PneumoniaMNIST class counts (minority = normal)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "class_counts.png"), dpi=150)
    plt.close(fig)

    # ---- 16 examples per class
    tr = splits["train"]
    fig, axes = plt.subplots(4, 8, figsize=(9, 5))
    g = torch.Generator().manual_seed(args.seed)
    for c in (0, 1):
        idx = torch.nonzero(tr.y == c).squeeze(1)
        pick = idx[torch.randperm(len(idx), generator=g)[:16]]
        for k, j in enumerate(pick):
            ax = axes[2 * c + k // 8, k % 8]
            ax.imshow(tr.x28[j, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if k % 8 == 0:
                ax.set_ylabel(["normal", "pneumonia"][c], fontsize=9)
    fig.suptitle("Real training images (16 per class, 28x28)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "examples_per_class.png"), dpi=150)
    plt.close(fig)

    # ---- mean image + intensity histogram per class
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for c in (0, 1):
        axes[c].imshow(tr.x28[tr.y == c].mean(0)[0], cmap="gray", vmin=0, vmax=1)
        axes[c].set_title(f"mean image: {['normal', 'pneumonia'][c]}")
        axes[c].axis("off")
        axes[2].hist(tr.x28[tr.y == c].flatten().numpy(), bins=50, alpha=0.6, density=True,
                     label=["normal", "pneumonia"][c])
    axes[2].set_title("pixel intensity distribution")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "class_means_hist.png"), dpi=150)
    plt.close(fig)
    print(f"Saved figures to {args.out}")


if __name__ == "__main__":
    main()
