"""Steps 4-5 - distributed hyper-parameter search for the conditional GAN with Ray Tune.

Smoke test : python tune_gan.py --num-samples 2 --max-epochs 2 --smoke
Full search: python tune_gan.py --num-samples 24 --max-epochs 40 --gpus-per-trial 0.25

Ray features used
* search space with tune.loguniform / tune.choice (reproducible random search, seeded)
* ASHAScheduler early-stops weak trials on `quality_score`
* per-trial resources via tune.with_resources (fractional GPUs -> several trials share
  one A100 without oversubscription; Ray enforces the accounting)
* tune.with_parameters ships the in-memory dataset + judge weights once via the object store
* checkpoint reporting every iteration, CheckpointConfig keeps the best 2 per trial,
  get_checkpoint() makes trials resumable after failures (FailureConfig)
* ResultGrid analysis -> CSV, curves, best config, best checkpoint export
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
import time

import torch

from gan_lab.data import load_pneumoniamnist, to_gan_space
from gan_lab.utils import save_json


def gan_trainable(config: dict, splits=None, judge_state=None):
    """Executed inside a Ray worker; one call = one trial."""
    from gan_lab.metrics import GANEvaluator
    from gan_lab.models import generate, load_judge
    from gan_lab.ray_compat import Checkpoint, get_checkpoint, report
    from gan_lab.trainer import DEFAULT_CONFIG, GANTrainer
    from gan_lab.utils import get_device, save_grid, set_seed

    cfg = {**DEFAULT_CONFIG, **config}
    if "ndf" not in config:  # tie discriminator width to generator width
        cfg["ndf"] = cfg["ngf"]
    set_seed(int(cfg["seed"]))
    device = get_device()
    trainer = GANTrainer(cfg, device)
    evaluator = GANEvaluator(splits, load_judge(judge_state, device), device, img_size=cfg["img_size"],
                             n_per_class=cfg["eval_n"], clf_epochs=cfg["clf_epochs"],
                             clf_seeds=tuple(range(cfg["clf_seeds"])))

    ckpt = get_checkpoint()
    if ckpt:  # resume (fault tolerance / PBT-style restarts)
        with ckpt.as_directory() as d:
            trainer.load_state_dict(torch.load(os.path.join(d, "checkpoint.pt"), map_location=device,
                                               weights_only=False))

    x = to_gan_space(splits["train"].x28, cfg["img_size"]).to(device)
    y = splits["train"].y.to(device)
    epr = int(cfg["epochs_per_report"])
    t0 = time.time()
    while trainer.epoch < cfg["max_epochs"]:
        for _ in range(epr):
            stats = trainer.train_epoch(x, y)
        metrics = {**stats, **evaluator.evaluate(trainer.G, downstream=True),
                   "epoch": trainer.epoch, "time_total_s_trial": time.time() - t0}
        if not all(math.isfinite(v) for v in metrics.values() if isinstance(v, float)):
            metrics = {k: (v if isinstance(v, (int, float)) and math.isfinite(v) else -1.0)
                       for k, v in metrics.items()}
            metrics["quality_score"] = -1.0
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(trainer.state_dict(), os.path.join(tmp, "checkpoint.pt"))
            trainer.G.eval()
            save_grid(generate(trainer.G, [0] * 8 + [1] * 8, seed=999, device=device),
                      os.path.join(tmp, "grid.png"), nrow=8)
            trainer.G.train()
            report(metrics, checkpoint=Checkpoint.from_directory(tmp))


class ResourceMonitor:
    """Samples Ray's cluster-wide resource accounting while Tune runs (a text stand-in for
    the dashboard when port 8265 is not reachable, e.g. on Colab)."""

    def __init__(self, path: str, every_s: float = 10.0):
        import threading

        self.path, self.every_s = path, every_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import ray

        t0 = time.time()
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("t_s,cpu_total,cpu_used,gpu_total,gpu_used,gpu_util_pct,gpu_mem_used_mb\n")
            while not self._stop.is_set():
                tot, av = ray.cluster_resources(), ray.available_resources()
                util, mem = "", ""
                try:
                    import subprocess
                    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                       timeout=5).stdout.strip().splitlines()[0].split(",")
                    util, mem = q[0].strip(), q[1].strip()
                except Exception:
                    pass
                f.write(f"{time.time() - t0:.0f},{tot.get('CPU', 0)},{tot.get('CPU', 0) - av.get('CPU', 0)},"
                        f"{tot.get('GPU', 0)},{tot.get('GPU', 0) - av.get('GPU', 0)},{util},{mem}\n")
                f.flush()
                self._stop.wait(self.every_s)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=15)


SEARCH_KEYS = ["lr_g", "lr_d", "beta1", "latent_dim", "ngf", "label_smoothing", "d_steps",
               "spectral_norm", "batch_size"]


def build_search_space(args):
    from ray import tune

    space = {
        "lr_g": tune.loguniform(1e-5, 5e-4),
        "lr_d": tune.loguniform(1e-5, 5e-4),
        "beta1": tune.choice([0.0, 0.5]),
        "latent_dim": tune.choice([32, 64, 128]),
        "ngf": tune.choice([32, 64]),
        "label_smoothing": tune.choice([0.0, 0.1]),
        "d_steps": tune.choice([1, 2]),
        "spectral_norm": tune.choice([False, True]),
        "batch_size": tune.choice([64, 128]),
        # constants passed through config
        "img_size": args.img_size,
        "max_epochs": args.max_epochs,
        "epochs_per_report": args.epochs_per_report,
        "eval_n": 200 if args.smoke else 500,
        "clf_epochs": 2 if args.smoke else 4,
        "clf_seeds": 1 if args.smoke else 2,
        "seed": args.seed,
    }
    return space


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=24)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--epochs-per-report", type=int, default=2, help="training epochs per Tune iteration")
    ap.add_argument("--grace-period", type=int, default=None, help="ASHA grace period (iterations)")
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpus-per-trial", type=float, default=2)
    ap.add_argument("--gpus-per-trial", type=float, default=None, help="default 0.25 if CUDA else 0")
    ap.add_argument("--max-concurrent", type=int, default=0)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--judge", default="artifacts/judge.pt")
    ap.add_argument("--storage-path", default="ray_results")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None, help="analysis output dir (default outputs/tune_<name>)")
    ap.add_argument("--export", default="artifacts/best_generator.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--address", default=None, help="Ray cluster address (None = start local)")
    args = ap.parse_args()
    if args.smoke:
        args.epochs_per_report = 1
    name = args.name or ("cgan_smoke" if args.smoke else "cgan_search")
    out = args.out or os.path.join("outputs", f"tune_{name}")
    os.makedirs(out, exist_ok=True)

    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from gan_lab.ray_compat import CheckpointConfig, FailureConfig, RunConfig

    project_dir = os.path.dirname(os.path.abspath(__file__))
    if not ray.is_initialized():
        ray.init(address=args.address, ignore_reinit_error=True,
                 runtime_env={"env_vars": {"PYTHONPATH": project_dir}})
    print("Ray cluster resources:", ray.cluster_resources())

    gpus = args.gpus_per_trial
    if gpus is None:
        gpus = 0.25 if torch.cuda.is_available() else 0.0
    resources = {"cpu": args.cpus_per_trial, "gpu": gpus}
    print("Per-trial resources:", resources)

    splits = load_pneumoniamnist(args.data_root)
    if not os.path.exists(args.judge):
        raise SystemExit(f"judge checkpoint missing: {args.judge} (run train_judge.py first)")
    judge_state = torch.load(args.judge, map_location="cpu", weights_only=False)["state_dict"]

    max_t = max(1, args.max_epochs // args.epochs_per_report)
    grace = args.grace_period or (1 if args.smoke else max(1, min(3, max_t)))
    scheduler = ASHAScheduler(time_attr="training_iteration", max_t=max_t, grace_period=grace,
                              reduction_factor=2)
    try:
        from ray.tune.search import BasicVariantGenerator
        search_alg = BasicVariantGenerator(random_state=args.seed)
    except TypeError:  # very old Ray
        search_alg = None

    trainable = tune.with_resources(tune.with_parameters(gan_trainable, splits=splits, judge_state=judge_state),
                                    resources)
    tuner = tune.Tuner(
        trainable,
        param_space=build_search_space(args),
        tune_config=tune.TuneConfig(metric="quality_score", mode="max", scheduler=scheduler,
                                    num_samples=args.num_samples, search_alg=search_alg,
                                    max_concurrent_trials=args.max_concurrent or None),
        run_config=RunConfig(
            name=name, storage_path=os.path.abspath(args.storage_path),
            checkpoint_config=CheckpointConfig(num_to_keep=2, checkpoint_score_attribute="quality_score",
                                               checkpoint_score_order="max"),
            failure_config=FailureConfig(max_failures=1),
        ),
    )
    monitor = ResourceMonitor(os.path.join(out, "resource_log.csv"))
    monitor.start()
    t0 = time.time()
    try:
        results = tuner.fit()
    finally:
        monitor.stop()
    wall = time.time() - t0
    print(f"Tune finished in {wall / 60:.1f} min; errors: {results.num_errors}")
    analyse(results, args, out, wall, resources, max_t, grace)


def analyse(results, args, out, wall, resources, max_t, grace):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from gan_lab.models import Generator, save_generator

    # one row per trial: last reported + best-ever quality score
    rows, curves = [], {}
    for r in results:
        if r.error or r.metrics is None:
            continue
        hist = r.metrics_dataframe
        tid = r.metrics.get("trial_id", os.path.basename(r.path))
        curves[tid] = hist
        row = {"trial_id": tid, **{k: r.config.get(k) for k in SEARCH_KEYS},
               "iterations": int(hist["training_iteration"].max()),
               "epochs": int(hist["epoch"].max()),
               "stopped_early": int(hist["training_iteration"].max()) < max_t,
               "best_quality_score": float(hist["quality_score"].max()),
               "final_quality_score": float(hist["quality_score"].iloc[-1])}
        best_i = hist["quality_score"].idxmax()
        for k in ["val_f1_gain", "collapse_penalty", "memorization_penalty", "fd_mean", "class_consistency",
                  "div_ratio_0", "div_ratio_1", "mem_rate", "d_acc"]:
            if k in hist:
                row[f"{k}@best"] = float(hist.loc[best_i, k])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("best_quality_score", ascending=False)
    df.to_csv(os.path.join(out, "trials.csv"), index=False)
    pd.concat([c.assign(trial_id=t) for t, c in curves.items()]).to_csv(os.path.join(out, "all_iterations.csv"),
                                                                       index=False)

    best = results.get_best_result(metric="quality_score", mode="max", scope="all")
    best_ckpt = best.get_best_checkpoint(metric="quality_score", mode="max")
    with best_ckpt.as_directory() as d:
        state = torch.load(os.path.join(d, "checkpoint.pt"), map_location="cpu", weights_only=False)
        if os.path.exists(os.path.join(d, "grid.png")):
            shutil.copy(os.path.join(d, "grid.png"), os.path.join(out, "best_checkpoint_grid.png"))
    cfg = state["config"]
    G = Generator(latent_dim=cfg["latent_dim"], ngf=cfg["ngf"], img_size=cfg["img_size"],
                  embed_dim=cfg.get("embed_dim", 32))
    G.load_state_dict(state["G"])
    bh = best.metrics_dataframe
    best_row = bh.loc[bh["quality_score"].idxmax()].to_dict()
    meta = {"source": "ray_tune", "experiment": os.path.basename(best.path), "trial_path": best.path,
            "selected_epoch": state["epoch"], "selection_metric": "quality_score (max over all reports)",
            "metrics_at_selection": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                                     for k, v in best_row.items() if not str(k).startswith("config/")}}
    os.makedirs(os.path.dirname(args.export) or ".", exist_ok=True)
    save_generator(args.export, G, cfg, meta)
    torch.save(state, os.path.splitext(args.export)[0] + "_full_checkpoint.pt")
    summary = {"wall_clock_min": wall / 60, "num_trials": len(df), "num_errors": results.num_errors,
               "num_stopped_early": int(df["stopped_early"].sum()) if len(df) else 0,
               "asha": {"max_t": max_t, "grace_period": grace, "reduction_factor": 2,
                        "epochs_per_iteration": args.epochs_per_report},
               "resources_per_trial": resources, "best_config": {k: cfg[k] for k in cfg},
               "best_selected_epoch": state["epoch"], "best_metrics": meta["metrics_at_selection"],
               "export": args.export}
    save_json(summary, os.path.join(out, "tune_summary.json"))
    print("Best config:", {k: cfg[k] for k in SEARCH_KEYS})
    print("Best metrics:", {k: round(v, 4) for k, v in meta["metrics_at_selection"].items()
                            if isinstance(v, float)})
    print(df.head(10).to_string())

    # ---- figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for t, c in curves.items():
        lw, alpha = (2.5, 1.0) if t == best_row.get("trial_id") else (1.0, 0.45)
        axes[0].plot(c["epoch"], c["quality_score"], lw=lw, alpha=alpha)
        axes[1].plot(c["epoch"], c["fd_mean"], lw=lw, alpha=alpha)
        axes[2].plot(c["epoch"], c["d_acc"], lw=lw, alpha=alpha)
    axes[0].set_title("quality_score per trial (ASHA stops weak ones early)")
    axes[1].set_title("Frechet distance (judge features)")
    axes[2].set_title("discriminator accuracy")
    for a in axes:
        a.set_xlabel("epoch")
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "trial_curves.png"), dpi=150)
    plt.close(fig)

    keys = [k for k in SEARCH_KEYS if k in df]
    fig, axes = plt.subplots(2, math.ceil(len(keys) / 2), figsize=(15, 6))
    for ax, k in zip(axes.flat, keys):
        vals = df[k]
        if k in ("lr_g", "lr_d"):
            ax.scatter(vals, df["best_quality_score"], c=df["stopped_early"].map({True: "#9D9D9D", False: "#E45756"}))
            ax.set_xscale("log")
        else:
            cats = sorted(vals.astype(str).unique())
            for i, cat in enumerate(cats):
                sub = df[vals.astype(str) == cat]
                ax.scatter([i] * len(sub), sub["best_quality_score"],
                           c=sub["stopped_early"].map({True: "#9D9D9D", False: "#E45756"}))
            ax.set_xticks(range(len(cats)), cats)
        ax.set_title(k)
        ax.grid(alpha=0.25)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.suptitle("best quality_score vs hyper-parameter (red = ran to max_t, grey = stopped by ASHA)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "hparam_scatter.png"), dpi=150)
    plt.close(fig)
    rl = os.path.join(out, "resource_log.csv")
    if os.path.exists(rl):
        r = pd.read_csv(rl)
        if len(r):
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.step(r["t_s"] / 60, r["cpu_used"], where="post", label=f"CPUs reserved (of {r['cpu_total'].max():g})")
            ax.step(r["t_s"] / 60, r["gpu_used"] * 4, where="post",
                    label=f"GPU reserved x4 (of {r['gpu_total'].max():g} GPU)")
            ax.set_xlabel("minutes")
            ax.set_title("Ray resource reservations during the Tune run")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "resource_usage.png"), dpi=150)
            plt.close(fig)
    print("analysis saved to", out, "| best generator exported to", args.export)


if __name__ == "__main__":
    main()
