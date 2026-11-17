"""Steps 3 + 6 - evaluate and stress-test the selected generator.

python evaluate.py --checkpoint artifacts/best_generator.pt --out outputs/eval --seeds 5

Produces
* fixed-seed grids per class, real-vs-fake panel, latent interpolation, same-z class swap
* quantitative GAN metrics (FD, diversity, class consistency, memorisation) - metrics.json
* nearest-neighbour memorisation figure + NN distance histogram
* failure cases: class-inconsistent samples and closest-to-training samples
* downstream classifier comparison (multiple seeds, held-out TEST set):
  real-only | class-weighted | oversample+naive aug | real+synthetic (0.5x, 1x gap)
  | synthetic-only (train-on-synthetic, test-on-real)
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from gan_lab import LABEL_NAMES
from gan_lab.classifier import compute_metrics, inverse_frequency_weights, predict_proba, train_classifier
from gan_lab.data import class_counts, from_gan_space, load_pneumoniamnist, roundtrip
from gan_lab.metrics import GANEvaluator, judge_outputs, nn_distances
from gan_lab.models import generate, load_generator, load_judge, sample_latent
from gan_lab.plots import BLUE, GREY, RED
from gan_lab.utils import save_grid, save_json, set_seed, get_device

METRICS = ["auc", "f1_macro", "f1_normal", "balanced_accuracy"]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                        for v in r.values) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------- qualitative

def qualitative(G, cfg, splits, device, out):
    for c in (0, 1):
        x = generate(G, [c] * 64, seed=1234 + c, device=device)
        save_grid(x, os.path.join(out, f"grid_{LABEL_NAMES[c]}_seed{1234 + c}.png"), nrow=8)

    # real vs fake (28x28 space, same display)
    tr = splits["train"]
    g = torch.Generator().manual_seed(0)
    fig, axes = plt.subplots(4, 8, figsize=(9, 5))
    for c in (0, 1):
        idx = torch.nonzero(tr.y == c).squeeze(1)
        real = tr.x28[idx[torch.randperm(len(idx), generator=g)[:8]]]
        fake = from_gan_space(generate(G, [c] * 8, seed=77 + c, device=device))
        for k in range(8):
            axes[2 * c, k].imshow(real[k, 0], cmap="gray", vmin=0, vmax=1)
            axes[2 * c + 1, k].imshow(fake[k, 0], cmap="gray", vmin=0, vmax=1)
        axes[2 * c, 0].set_ylabel(f"REAL\n{LABEL_NAMES[c]}", fontsize=8)
        axes[2 * c + 1, 0].set_ylabel(f"SYNTHETIC\n{LABEL_NAMES[c]}", fontsize=8)
    for a in axes.flat:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Real (rows 1,3) vs synthetic (rows 2,4) at 28x28")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "real_vs_synthetic.png"), dpi=150)
    plt.close(fig)

    # latent interpolation per class + same z with label swapped
    steps = 10
    z = sample_latent(4, G.latent_dim, seed=2024)
    rows, labels = [], []
    for c in (0, 1):
        for pair in ((0, 1), (2, 3)):
            a = torch.linspace(0, 1, steps)[:, None]
            zz = (1 - a) * z[pair[0]] + a * z[pair[1]]
            rows.append(zz)
            labels += [c] * steps
    zs = torch.cat(rows).to(device)
    with torch.no_grad():
        imgs = G(zs, torch.tensor(labels, device=device)).cpu()
    save_grid(imgs, os.path.join(out, "latent_interpolation.png"), nrow=steps)

    zsw = sample_latent(10, G.latent_dim, seed=31).to(device)
    with torch.no_grad():
        a = G(zsw, torch.zeros(10, dtype=torch.long, device=device)).cpu()
        b = G(zsw, torch.ones(10, dtype=torch.long, device=device)).cpu()
    save_grid(torch.cat([a, b]), os.path.join(out, "class_swap_same_z.png"), nrow=10)


# --------------------------------------------------------------- memorisation + failures

def memorisation_and_failures(G, cfg, splits, judge, ev: GANEvaluator, device, out, n=1000):
    labels = torch.tensor([0] * n + [1] * n)
    fake = ev.generate28(G, labels, seed=4321)
    fake_nn, fake_idx = nn_distances(fake.to(device), ev.train_flat)
    val_nn = ev.val_nn
    thr = float(torch.quantile(val_nn, 0.05))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(0, float(max(fake_nn.max(), val_nn.max())), 60)
    ax.hist(val_nn.numpy(), bins=bins, density=True, alpha=0.6, color=GREY, label="real val -> train")
    ax.hist(fake_nn.numpy(), bins=bins, density=True, alpha=0.6, color=RED, label="synthetic -> train")
    ax.axvline(thr, color="black", ls="--", lw=1, label="5th pct of val (flag threshold)")
    ax.set_xlabel("nearest-neighbour RMSE to training set (28x28, [0,1])")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "nn_distance_hist.png"), dpi=150)
    plt.close(fig)

    # closest synthetic samples with their 3 nearest training images
    order = torch.argsort(fake_nn)[:8]
    trf = ev.train_flat
    fig, axes = plt.subplots(8, 4, figsize=(4.6, 9))
    for r, i in enumerate(order):
        d = torch.cdist(fake[i].flatten()[None].to(device), trf)[0]
        nn3 = torch.argsort(d)[:3].cpu()
        axes[r, 0].imshow(fake[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(f"{LABEL_NAMES[int(labels[i])][:4]}\n{fake_nn[i]:.3f}", fontsize=7)
        for k, j in enumerate(nn3):
            axes[r, k + 1].imshow(ev.x_tr[j, 0], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("synthetic", fontsize=8)
    for k in range(3):
        axes[0, k + 1].set_title(f"train NN{k + 1}", fontsize=8)
    for a in axes.flat:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Closest synthetic samples to training data", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "memorization_nearest_neighbors.png"), dpi=150)
    plt.close(fig)

    # class-inconsistent samples (judge disagrees with requested label, most confident first)
    _, probs = judge_outputs(judge, fake, device)
    wrong_conf = probs[torch.arange(len(labels)), 1 - labels]
    fig, axes = plt.subplots(2, 8, figsize=(9, 2.8))
    n_wrong = {}
    for c in (0, 1):
        m = (labels == c) & (probs.argmax(1) != c)
        n_wrong[LABEL_NAMES[c]] = int(m.sum())
        idx = torch.nonzero(m).squeeze(1)
        idx = idx[torch.argsort(wrong_conf[idx], descending=True)][:8]
        for k in range(8):
            ax = axes[c, k]
            ax.set_xticks([])
            ax.set_yticks([])
            if k < len(idx):
                ax.imshow(fake[idx[k], 0], cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"p={wrong_conf[idx[k]]:.2f}", fontsize=7)
        axes[c, 0].set_ylabel(f"asked:\n{LABEL_NAMES[c]}", fontsize=8)
    fig.suptitle("Failure cases: judge confidently assigns the OTHER class", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "failure_class_inconsistent.png"), dpi=150)
    plt.close(fig)

    exact_dupes = int((fake_nn < 1e-3).sum())
    return {"n_class_inconsistent": n_wrong, "n_near_exact_duplicates": exact_dupes,
            "closest_nn_rmse": float(fake_nn.min()), "flag_threshold": thr,
            "flag_rate": float((fake_nn < thr).float().mean())}


# ------------------------------------------------------------------------- downstream

def downstream(G, splits, device, out, seeds, epochs, img_size):
    rt = {k: (roundtrip(v.x28, img_size), v.y) for k, v in splits.items()}
    xtr, ytr = rt["train"]
    counts = class_counts(ytr)
    minority = int(np.argmin(counts))
    gap = max(counts) - min(counts)
    rows, cms = [], {}

    def run(name, x, y, seed, **kw):
        model, info = train_classifier(x, y, device, epochs=epochs, seed=seed,
                                       x_val=rt["val"][0], y_val=rt["val"][1], **kw)
        m = compute_metrics(rt["test"][1].numpy(), predict_proba(model, rt["test"][0], device))
        cms.setdefault(name, np.zeros((2, 2), dtype=int))
        cms[name] += np.array(m["confusion_matrix"])
        rows.append({"condition": name, "seed": seed, "best_val_epoch": info["best_epoch"],
                     "n_train": len(y), **{k: m[k] for k in METRICS + ["f1_pneumonia", "accuracy"]}})
        print(f"  {name:22s} seed={seed} " + " ".join(f"{k}={m[k]:.3f}" for k in METRICS), flush=True)

    for seed in seeds:
        print(f"[downstream] seed {seed}")
        run("real_only", xtr, ytr, seed)
        run("real_class_weighted", xtr, ytr, seed, class_weight=inverse_frequency_weights(ytr))
        g = torch.Generator().manual_seed(seed)
        mi = torch.nonzero(ytr == minority).squeeze(1)
        extra = mi[torch.randint(len(mi), (gap,), generator=g)]
        run("real_oversample_aug", torch.cat([xtr, xtr[extra]]), torch.cat([ytr, ytr[extra]]), seed, augment=True)
        for frac in (0.5, 1.0):
            k = int(round(frac * gap))
            syn = from_gan_space(generate(G, [minority] * k, seed=10_000 + seed, device=device))
            run(f"real_plus_synth_{frac:g}x", torch.cat([xtr, syn]),
                torch.cat([ytr, torch.full((k,), minority, dtype=torch.long)]), seed)
        half = len(ytr) // 2
        syn_all = from_gan_space(generate(G, [0] * half + [1] * half, seed=20_000 + seed, device=device))
        run("synthetic_only_TSTR", syn_all, torch.tensor([0] * half + [1] * half), seed)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "downstream_per_seed.csv"), index=False)
    order = list(dict.fromkeys(df["condition"]))
    agg = df.groupby("condition", sort=False)[METRICS].agg(["mean", "std"])
    summary = pd.DataFrame({"condition": order})
    for mt in METRICS:
        summary[mt] = [f"{agg.loc[c, (mt, 'mean')]:.4f} ± {agg.loc[c, (mt, 'std')]:.4f}" for c in order]

    # paired comparison vs real_only (same seeds = same init + data order)
    base = df[df.condition == "real_only"].set_index("seed")
    paired = []
    for c in order:
        if c == "real_only":
            continue
        other = df[df.condition == c].set_index("seed").loc[base.index]
        for mt in METRICS:
            d = other[mt].values - base[mt].values
            n = len(d)
            se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
            p = stats.ttest_rel(other[mt].values, base[mt].values).pvalue if n > 1 else np.nan
            paired.append({"condition": c, "metric": mt, "mean_delta": float(d.mean()),
                           "ci95_low": float(d.mean() - tcrit * se), "ci95_high": float(d.mean() + tcrit * se),
                           "p_value_paired_t": float(p), "wins": f"{int((d > 0).sum())}/{n}"})
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(os.path.join(out, "downstream_paired_vs_real_only.csv"), index=False)

    with open(os.path.join(out, "downstream_tables.md"), "w", encoding="utf-8") as f:
        f.write(f"### Test-set metrics (mean ± std over {len(seeds)} seeds)\n\n")
        f.write(md_table(summary) + "\n\n### Paired difference vs real_only\n\n")
        f.write(md_table(paired_df) + "\n")

    # bar chart
    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 3.8), sharey=False)
    colors = [GREY, "#BAB0AC", "#B279A2", BLUE, "#72B7B2", RED]
    for ax, mt in zip(axes, METRICS):
        means = [agg.loc[c, (mt, "mean")] for c in order]
        sds = [agg.loc[c, (mt, "std")] for c in order]
        ax.bar(range(len(order)), means, yerr=sds, color=colors[:len(order)], capsize=3)
        lo = min(m - s for m, s in zip(means, sds))
        ax.set_ylim(max(0, lo - 0.05), min(1.0, max(means) + 0.05))
        ax.set_xticks(range(len(order)), [o.replace("_", "\n") for o in order], fontsize=6)
        ax.set_title(mt)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Downstream classifier on held-out TEST set (mean ± sd, {len(seeds)} seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "downstream_bars.png"), dpi=150)
    plt.close(fig)

    # confusion matrices (summed over seeds)
    show = [c for c in ("real_only", "real_class_weighted", "real_plus_synth_1x", "synthetic_only_TSTR") if c in cms]
    fig, axes = plt.subplots(1, len(show), figsize=(3.2 * len(show), 3))
    for ax, c in zip(np.atleast_1d(axes), show):
        cm = cms[c]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["normal", "pneum."])
        ax.set_yticks([0, 1], ["normal", "pneum."])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(c, fontsize=8)
    fig.suptitle("Test confusion matrices (summed over seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion_matrices.png"), dpi=150)
    plt.close(fig)

    return {
        "summary_table": summary.to_dict(orient="records"),
        "paired": paired_df.to_dict(orient="records"),
        "means": {c: {mt: float(agg.loc[c, (mt, "mean")]) for mt in METRICS} for c in order},
        "stds": {c: {mt: float(agg.loc[c, (mt, "std")]) for mt in METRICS} for c in order},
        "confusion_matrices": {k: v.tolist() for k, v in cms.items()},
        "minority_label": minority, "gap": gap, "seeds": list(seeds), "epochs": epochs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/best_generator.pt")
    ap.add_argument("--judge", default="artifacts/judge.pt")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--clf-epochs", type=int, default=20)
    ap.add_argument("--skip-downstream", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(0)
    device = get_device()

    splits = load_pneumoniamnist(args.data_root)
    G, cfg, meta = load_generator(args.checkpoint, device)
    judge = load_judge(args.judge, device)
    print("loaded generator", cfg, "meta keys:", list(meta))

    qualitative(G, cfg, splits, device, args.out)
    ev = GANEvaluator(splits, judge, device, img_size=cfg["img_size"], n_per_class=1000,
                      clf_epochs=4, clf_seeds=(0, 1, 2))
    gan_metrics = ev.evaluate(G, downstream=True)
    print("GAN metrics:", {k: round(v, 4) for k, v in gan_metrics.items()})
    mem = memorisation_and_failures(G, cfg, splits, judge, ev, device, args.out)
    print("memorisation/failures:", mem)

    result = {"config": cfg, "checkpoint_meta": meta, "gan_metrics": gan_metrics, "memorization": mem}
    if not args.skip_downstream:
        result["downstream"] = downstream(G, splits, device, args.out, list(range(args.seeds)),
                                          args.clf_epochs, cfg["img_size"])
        print(open(os.path.join(args.out, "downstream_tables.md"), encoding="utf-8").read())
    save_json(result, os.path.join(args.out, "eval_summary.json"))
    print("saved", os.path.join(args.out, "eval_summary.json"))


if __name__ == "__main__":
    main()
