"""One-command pipeline for a local GPU machine (tuned for an RTX 5090, 32 GB).

python run_pipeline.py --check          # preflight only: GPU, torch build, Ray
python run_pipeline.py --smoke          # fast end-to-end sanity run (~10-20 min on a 5090)
python run_pipeline.py                  # full study
python run_pipeline.py --only tune evaluate   # run selected stages
python run_pipeline.py --skip serve     # skip stages

Stages (in order): tests data judge baseline unstable tune evaluate evaluate_baseline
                   repro serve report
Each stage is a normal script call, so any of them can be re-run on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

STAGES = ["tests", "data", "judge", "baseline", "unstable", "tune", "evaluate",
          "evaluate_baseline", "repro", "serve", "report"]
ROOT = os.path.dirname(os.path.abspath(__file__))


def preflight() -> dict:
    import torch

    info = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "cpus": os.cpu_count() or 1}
    if not torch.cuda.is_available():
        print("[preflight] CUDA is NOT available. Training would run on CPU (very slow).")
        print("  RTX 50-series (Blackwell, sm_120) needs a CUDA 12.8+ build of PyTorch:")
        print("  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        return info
    p = torch.cuda.get_device_properties(0)
    info.update(gpu=p.name, vram_gb=round(p.total_memory / 2**30, 1), capability=f"{p.major}.{p.minor}",
                arch_list=torch.cuda.get_arch_list())
    print(f"[preflight] {p.name} | {info['vram_gb']} GB | sm_{p.major}{p.minor} | torch {torch.__version__} "
          f"(CUDA {torch.version.cuda})")
    cap = f"sm_{p.major}{p.minor}"
    archs = info["arch_list"]
    if cap not in archs and not any(a.startswith("compute_") and a >= f"compute_{p.major}{p.minor}" for a in archs):
        print(f"[preflight] WARNING: this torch build ships kernels for {archs} but the GPU is {cap}.")
        print("  Install a CUDA 12.8+ build: pip install --upgrade torch torchvision "
              "--index-url https://download.pytorch.org/whl/cu128")
    try:  # actually run a conv + matmul on the GPU: catches 'no kernel image' errors early
        x = torch.randn(8, 1, 64, 64, device="cuda")
        y = torch.nn.functional.conv2d(x, torch.randn(8, 1, 3, 3, device="cuda"))
        (y.flatten(1) @ torch.randn(y[0].numel(), 4, device="cuda")).sum().item()
        print("[preflight] CUDA kernels OK")
    except Exception as e:  # noqa: BLE001
        print("[preflight] CUDA kernel test FAILED:", e)
        info["kernel_ok"] = False
    return info


def profile(info: dict, smoke: bool) -> dict:
    """Run sizes from the hardware. A 5090 (32 GB) is far larger than these tiny GANs need, so we
    use it by running many Ray trials concurrently, each reserving a fraction of the GPU."""
    cpus = info["cpus"]
    big = info.get("vram_gb", 0) >= 20
    conc = 5 if big else 4                       # concurrent Tune trials
    cpt = max(1, min(4, cpus // conc)) if cpus >= conc else cpus / conc
    p = {
        "tune_samples": 32 if big else 16, "tune_epochs": 40 if big else 30,
        "base_epochs": 60, "unstable_epochs": 40, "gpus_per_trial": round(1.0 / conc, 3),
        "cpus_per_trial": cpt, "serve_cpus": 1 if cpus >= 8 else 0.4, "serve_gpus": 0.1 if big else 0.25,
        "serve_max_replicas": 4, "eval_seeds": 5, "clf_epochs": 20,
        "load_levels": ["1", "4", "16", "32", "64"], "load_duration": "20",
    }
    if smoke:
        p.update(tune_samples=2, tune_epochs=2, base_epochs=3, unstable_epochs=3, eval_seeds=1, clf_epochs=2,
                 load_levels=["1", "8"], load_duration="5")
    return p


def run(cmd: list[str], name: str, log_dir: str) -> float:
    os.makedirs(log_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n=== [{name}] {' '.join(cmd)}", flush=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env={**os.environ, "PYTHONPATH": ROOT,
                                                                          "PYTHONIOENCODING": "utf-8"})
        for line in proc.stdout:
            log.write(line)
            low = line.lower()
            if "warn" not in low and line.strip():
                print("   ", line.rstrip()[:200], flush=True)
        rc = proc.wait()
    dt = time.time() - t0
    if rc != 0:
        raise SystemExit(f"stage '{name}' failed (exit {rc}); full log: {log_path}")
    print(f"=== [{name}] done in {dt / 60:.1f} min (log: {log_path})", flush=True)
    return dt


def serve_stage(p: dict, py: str) -> list[str]:
    """Deploy, exercise the API, load-test with replica tracing, shut down (single process)."""
    code = f"""
import sys, os
sys.path.insert(0, {ROOT!r})
import ray
from ray import serve
from serve_app import deploy, replica_count
deploy('artifacts/best_generator.pt', num_gpus={p['serve_gpus']}, num_cpus={p['serve_cpus']},
       max_replicas={p['serve_max_replicas']})
import subprocess
subprocess.run([sys.executable, 'client.py', '--url', 'http://127.0.0.1:8000', '--out', 'outputs/serve'], check=True)
from load_test import run_load_test
run_load_test('http://127.0.0.1:8000', levels=({', '.join(p['load_levels'])},), duration_s={p['load_duration']},
              n_images=8, probe=replica_count, out='outputs/serve')
serve.shutdown(); ray.shutdown()
"""
    return [py, "-c", code]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", choices=STAGES)
    ap.add_argument("--skip", nargs="*", choices=STAGES, default=[])
    args = ap.parse_args()

    info = preflight()
    if args.check:
        try:
            import ray
            print("[preflight] ray", ray.__version__)
        except ImportError:
            print("[preflight] ray not installed: pip install -r requirements.txt")
        return
    if not info.get("cuda_available"):
        print("[run] continuing on CPU (use --smoke for a quick check)")
    p = profile(info, args.smoke)
    print("[run] profile:", json.dumps(p))
    py = sys.executable
    logs = os.path.join("outputs", "logs" + ("_smoke" if args.smoke else ""))
    tune_name = "cgan_smoke" if args.smoke else "cgan_search"
    best = "artifacts/smoke_generator.pt" if args.smoke else "artifacts/best_generator.pt"
    seeds = str(p["eval_seeds"])

    cmds = {
        "tests": [py, "-m", "pytest", "-q"],
        "data": [py, "inspect_data.py"],
        "judge": [py, "train_judge.py", "--epochs", "3" if args.smoke else "25"],
        "baseline": [py, "train_gan.py", "--epochs", str(p["base_epochs"]), "--eval-every", "5" if not args.smoke else "1",
                     "--out", "runs/baseline"],
        "unstable": [py, "train_gan.py", "--epochs", str(p["unstable_epochs"]), "--eval-every", "5" if not args.smoke else "1",
                     "--out", "runs/unstable", "--set", "lr_d=1e-3", "lr_g=5e-5", "label_smoothing=0.0", "d_steps=2"],
        "tune": [py, "tune_gan.py", "--num-samples", str(p["tune_samples"]), "--max-epochs", str(p["tune_epochs"]),
                 "--epochs-per-report", "1" if args.smoke else "2", "--gpus-per-trial", str(p["gpus_per_trial"]),
                 "--cpus-per-trial", str(p["cpus_per_trial"]), "--export", best, "--name", tune_name]
                + (["--smoke"] if args.smoke else []),
        "evaluate": [py, "evaluate.py", "--checkpoint", best, "--out", "outputs/eval", "--seeds", seeds,
                     "--clf-epochs", str(p["clf_epochs"])],
        "evaluate_baseline": [py, "evaluate.py", "--checkpoint", "runs/baseline/generator.pt",
                              "--out", "outputs/eval_baseline", "--seeds", "1" if args.smoke else "3",
                              "--clf-epochs", str(p["clf_epochs"])],
        "repro": [py, "-c", (
            "import subprocess,sys,re,json\n"
            "def g(o):\n"
            f"    r=subprocess.run([sys.executable,'regenerate_grid.py','--checkpoint',{best!r},'--seed','1234','--out',o],"
            "capture_output=True,text=True).stdout\n"
            "    print(r); return re.search(r'images sha256=(\\w+)',r).group(1)\n"
            "a=g('outputs/fixed_grid_run1.png'); b=g('outputs/fixed_grid_run2.png')\n"
            "json.dump({'hash_run1':a,'hash_run2':b,'identical':a==b,'command':'python regenerate_grid.py "
            "--checkpoint artifacts/best_generator.pt --seed 1234'},open('outputs/repro.json','w'),indent=1)\n"
            "print('identical:',a==b)")],
        "serve": serve_stage(p, py),
        "report": [py, "make_report.py", "--out", "outputs/report"],
    }
    stages = args.only or [s for s in STAGES if s not in args.skip]
    timings = {}
    for s in stages:
        timings[s] = run(cmds[s], s, logs)
    print("\nStage timings (min):", {k: round(v / 60, 1) for k, v in timings.items()},
          "| total", round(sum(timings.values()) / 60, 1))
    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "timings.json" if not args.smoke else "timings_smoke.json"), "w") as f:
        json.dump({"hardware": info, "profile": p, "minutes": {k: v / 60 for k, v in timings.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
