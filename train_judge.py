"""Train the frozen 'judge' classifier on REAL data only.

It is used as the feature extractor for Frechet distance / diversity and as the class-
consistency checker for generated images. It never sees synthetic data.

python train_judge.py --out artifacts/judge.pt
"""
from __future__ import annotations

import argparse
import os

import torch

from gan_lab.classifier import compute_metrics, inverse_frequency_weights, predict_proba, train_classifier
from gan_lab.data import load_pneumoniamnist, roundtrip
from gan_lab.utils import get_device, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="artifacts/judge.pt")
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device()
    splits = load_pneumoniamnist(args.data_root)
    rt = {k: (roundtrip(v.x28, args.img_size), v.y) for k, v in splits.items()}
    # Class-weighted loss + light augmentation: the judge should be balanced, not a
    # majority-class detector, otherwise "class consistency" would be biased.
    model, info = train_classifier(*rt["train"], device, epochs=args.epochs, seed=args.seed,
                                   class_weight=inverse_frequency_weights(rt["train"][1]),
                                   augment=True, x_val=rt["val"][0], y_val=rt["val"][1])
    res = {"best_epoch": info["best_epoch"]}
    for s in ("val", "test"):
        res[s] = compute_metrics(rt[s][1].numpy(), predict_proba(model, rt[s][0], device))
        print(s, {k: round(v, 4) for k, v in res[s].items() if k != "confusion_matrix"},
              "CM", res[s]["confusion_matrix"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "img_size": args.img_size, "metrics": res}, args.out)
    save_json(res, os.path.splitext(args.out)[0] + "_metrics.json")
    print("saved", args.out)


if __name__ == "__main__":
    main()
