"""Reproducible command to regenerate a fixed image grid from a saved generator.

python regenerate_grid.py --checkpoint artifacts/best_generator.pt --seed 1234 --out outputs/fixed_grid.png

Prints a SHA-256 of the uint8 images: the same checkpoint + seed + device type gives the
same hash (the latent vectors come from a seeded CPU generator, so they are identical on
every machine; tiny float differences across GPU models can still flip a pixel value).
"""
from __future__ import annotations

import argparse

import torch

from gan_lab.models import generate, load_generator
from gan_lab.utils import array_sha256, file_sha256, get_device, save_grid, to_uint8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/best_generator.pt")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-per-class", type=int, default=16)
    ap.add_argument("--out", default="outputs/fixed_grid.png")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = get_device(args.device)
    G, cfg, meta = load_generator(args.checkpoint, device)
    labels = [0] * args.n_per_class + [1] * args.n_per_class
    x = generate(G, labels, seed=args.seed, device=device)
    save_grid(x, args.out, nrow=args.n_per_class if args.n_per_class <= 16 else 16)
    print(f"checkpoint sha256[:12]={file_sha256(args.checkpoint)[:12]} seed={args.seed} device={device.type}")
    print(f"rows: first {args.n_per_class} = normal, next {args.n_per_class} = pneumonia (SYNTHETIC)")
    print(f"images sha256={array_sha256(to_uint8(x))}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
