"""Step 8 - concurrent load test of the /generate endpoint + replica scaling trace.

python load_test.py --url http://127.0.0.1:8000 --levels 1 4 16 32 64 --duration 20

When run in the same Python process as Ray Serve (the notebook), pass
`probe=serve_app.replica_count` to record how many replicas are RUNNING over time.
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests


def _worker(url, payload, stop_at, out):
    s = requests.Session()
    while time.time() < stop_at:
        t0 = time.perf_counter()
        try:
            r = s.post(url, json=payload, timeout=60)
            ok = r.status_code == 200
        except requests.RequestException:
            ok = False
        out.append((time.time(), (time.perf_counter() - t0) * 1000, ok))


def run_level(url, concurrency, duration_s, n_images):
    results = []
    stop_at = time.time() + duration_s
    payload = {"label": 0, "n": n_images}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i in range(concurrency):
            p = dict(payload, label=i % 2)
            ex.submit(_worker, url, p, stop_at, results)
    lat = np.array([r[1] for r in results if r[2]])
    errors = sum(1 for r in results if not r[2])
    return {
        "concurrency": concurrency, "requests": len(results), "errors": errors,
        "throughput_rps": len(lat) / duration_s, "images_per_s": len(lat) * n_images / duration_s,
        "p50_ms": float(np.percentile(lat, 50)) if len(lat) else np.nan,
        "p95_ms": float(np.percentile(lat, 95)) if len(lat) else np.nan,
        "p99_ms": float(np.percentile(lat, 99)) if len(lat) else np.nan,
        "mean_ms": float(lat.mean()) if len(lat) else np.nan,
    }, results


def run_load_test(url="http://127.0.0.1:8000", levels=(1, 4, 16, 32, 64), duration_s=20, n_images=8,
                  probe=None, out="outputs/serve", cooldown_s=5):
    os.makedirs(out, exist_ok=True)
    endpoint = f"{url.rstrip('/')}/generate"
    requests.post(endpoint, json={"label": 0, "n": 1}, timeout=60)  # warm-up
    trace, stop = [], threading.Event()
    t_start = time.time()

    def poll():
        while not stop.is_set():
            try:
                st = probe() if probe else {}
                trace.append({"t_s": time.time() - t_start, "running": st.get("RUNNING", 0),
                              "starting": st.get("STARTING", 0)})
            except Exception as e:  # noqa: BLE001
                trace.append({"t_s": time.time() - t_start, "running": np.nan, "starting": np.nan, "err": str(e)})
            stop.wait(1.0)

    poller = threading.Thread(target=poll, daemon=True)
    if probe:
        poller.start()
    rows, phases = [], []
    try:
        for c in levels:
            t0 = time.time() - t_start
            summary, _ = run_level(endpoint, c, duration_s, n_images)
            phases.append((t0, time.time() - t_start, c))
            rows.append(summary)
            print(f"concurrency={c:3d} | {summary['throughput_rps']:7.1f} req/s | "
                  f"{summary['images_per_s']:8.1f} img/s | p50 {summary['p50_ms']:6.1f} ms | "
                  f"p95 {summary['p95_ms']:6.1f} ms | p99 {summary['p99_ms']:6.1f} ms | errors {summary['errors']}",
                  flush=True)
            time.sleep(cooldown_s)
        if probe:  # watch scale-down for a while
            time.sleep(45)
    finally:
        stop.set()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "load_test.csv"), index=False)
    tr = pd.DataFrame(trace)
    if len(tr):
        tr.to_csv(os.path.join(out, "replica_trace.csv"), index=False)
    plot(df, tr, phases, out)
    return df, tr


def plot(df, tr, phases, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3 if len(tr) else 2, figsize=(15 if len(tr) else 10, 3.8))
    ax = axes[0]
    ax.plot(df["concurrency"], df["p50_ms"], "o-", label="p50")
    ax.plot(df["concurrency"], df["p95_ms"], "o-", label="p95")
    ax.plot(df["concurrency"], df["p99_ms"], "o-", label="p99")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrent clients")
    ax.set_ylabel("latency (ms)")
    ax.set_title("latency vs concurrency")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.plot(df["concurrency"], df["throughput_rps"], "o-", color="#54A24B")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrent clients")
    ax.set_ylabel("requests / s")
    ax.set_title("throughput")
    if len(tr):
        ax = axes[2]
        ax.step(tr["t_s"], tr["running"], where="post", color="#E45756", label="RUNNING replicas")
        for t0, t1, c in phases:
            ax.axvspan(t0, t1, alpha=0.08, color="grey")
            ax.text((t0 + t1) / 2, 0.2, f"c={c}", ha="center", fontsize=7)
        ax.set_xlabel("seconds")
        ax.set_ylabel("replicas")
        ax.set_title("autoscaling trace")
        ax.set_ylim(0, max(4.5, float(np.nanmax(tr["running"])) + 0.5))
    for a in axes:
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "load_test.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4, 16, 32, 64])
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--n-images", type=int, default=8)
    ap.add_argument("--out", default="outputs/serve")
    ap.add_argument("--probe", action="store_true", help="connect to the running Ray cluster to trace replicas")
    args = ap.parse_args()
    probe = None
    if args.probe:
        import ray

        ray.init(address="auto", ignore_reinit_error=True)
        from serve_app import replica_count
        probe = replica_count
    run_load_test(args.url, args.levels, args.duration, args.n_images, probe, args.out)


if __name__ == "__main__":
    main()
