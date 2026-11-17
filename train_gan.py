"""Step 2 - train a single conditional DCGAN (baseline, or any config) with full logging.

python train_gan.py --epochs 60 --out runs/baseline
python train_gan.py --epochs 40 --out runs/unstable --set lr_d=1e-3 lr_g=5e-5 label_smoothing=0 d_steps=2

Logs: TensorBoard scalars + image grids (runs/<name>/tb), history.json, curves.png,
per-epoch fixed-noise grids, checkpoint.pt (full) and generator.pt (export).
"""
from __future__ import annotations

import argparse
import ast
import os
import time

import torch

from gan_lab.data import load_pneumoniamnist, to_gan_space
from gan_lab.metrics import GANEvaluator
from gan_lab.models import generate, load_judge, save_generator
from gan_lab.plots import plot_training_history
from gan_lab.trainer import DEFAULT_CONFIG, GANTrainer
from gan_lab.utils import get_device, save_grid, save_json, set_seed


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--judge", default="artifacts/judge.pt")
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. lr_d=1e-3")
    args = ap.parse_args()

    cfg = {**DEFAULT_CONFIG, **parse_overrides(args.set)}
    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(args.out, exist_ok=True)
    save_json(cfg, os.path.join(args.out, "config.json"))

    splits = load_pneumoniamnist(args.data_root)
    x = to_gan_space(splits["train"].x28, cfg["img_size"]).to(device)
    y = splits["train"].y.to(device)

    trainer = GANTrainer(cfg, device)
    print("params:", trainer.param_counts(), "device:", device)
    evaluator = None
    if os.path.exists(args.judge):
        evaluator = GANEvaluator(splits, load_judge(args.judge, device), device, img_size=cfg["img_size"])
    else:
        print(f"[warn] judge not found at {args.judge}; only adversarial metrics will be logged")

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.out, "tb"))
    except Exception as e:  # pragma: no cover
        print("[warn] TensorBoard unavailable:", e)
        writer = None

    fixed_labels = [0] * 8 + [1] * 8
    history = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        stats = trainer.train_epoch(x, y)
        row = {"epoch": ep, "time_s": time.time() - t0, **stats}
        if evaluator is not None and (ep % args.eval_every == 0 or ep == args.epochs or ep == 1):
            row.update(evaluator.evaluate(trainer.G, downstream=True))
        history.append(row)
        trainer.G.eval()
        grid = generate(trainer.G, fixed_labels, seed=999, device=device)
        trainer.G.train()
        save_grid(grid, os.path.join(args.out, "grids", f"epoch_{ep:03d}.png"), nrow=8)
        if writer:
            for k, v in row.items():
                if isinstance(v, (int, float)) and k != "epoch":
                    writer.add_scalar(k, v, ep)
            writer.add_images("fixed_grid(top=normal,bottom=pneumonia)", (grid + 1) / 2, ep)
        msg = (f"ep {ep:3d} | D {stats['d_loss']:.3f} G {stats['g_loss']:.3f} "
               f"Dacc {stats['d_acc']:.2f} D(x) {stats['d_real']:.2f} D(G(z)) {stats['d_fake']:.2f}")
        if "quality_score" in row:
            msg += (f" | FD {row['fd_mean']:.2f} div {row['div_ratio_0']:.2f}/{row['div_ratio_1']:.2f} "
                    f"cons {row['class_consistency']:.2f} mem {row['mem_rate']:.3f} "
                    f"dF1 {row['val_f1_gain']:+.3f} Q {row['quality_score']:+.3f}")
        print(msg, flush=True)

    if writer:
        writer.close()
    save_json(history, os.path.join(args.out, "history.json"))
    plot_training_history(history, os.path.join(args.out, "curves.png"), title=f"cGAN training: {args.out}")
    torch.save(trainer.state_dict(), os.path.join(args.out, "checkpoint.pt"))
    save_generator(os.path.join(args.out, "generator.pt"), trainer.G, trainer.cfg,
                   {"source": args.out, "epoch": args.epochs, "final_metrics": history[-1]})
    print(f"done in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
