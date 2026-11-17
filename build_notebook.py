"""Builds HW3a_ConditionalGAN.ipynb: a self-contained Colab notebook that writes the project
source with %%writefile (so it runs on a remote Colab runtime with no file upload) and then
executes every assignment step. Re-run after editing any source file:

python build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCES = [
    "pyproject.toml", "requirements.txt",
    "gan_lab/__init__.py", "gan_lab/utils.py", "gan_lab/data.py", "gan_lab/models.py", "gan_lab/trainer.py",
    "gan_lab/classifier.py", "gan_lab/metrics.py", "gan_lab/plots.py", "gan_lab/ray_compat.py", "gan_lab/service.py",
    "inspect_data.py", "train_judge.py", "train_gan.py", "tune_gan.py", "evaluate.py", "regenerate_grid.py",
    "serve_app.py", "client.py", "load_test.py", "make_report.py",
    "tests/test_models.py", "tests/test_metrics.py", "tests/test_service.py",
    "report/MODEL_CARD.md",
]

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s, collapsed=False):
    meta = {"collapsed": True, "jupyter": {"source_hidden": True}} if collapsed else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


SHOW = '''
from IPython.display import Image, Markdown, display
import json, os, pandas as pd
def show(*paths, width=None):
    for p in paths:
        if os.path.exists(p):
            display(Image(filename=p, width=width))
        else:
            print("missing:", p)
'''

md("""
# HW3a: Conditional GAN for synthetic PneumoniaMNIST + Ray Tune + Ray Serve

> **Safety boundary.** Every generated image here is **SYNTHETIC**. It is for education and research prototyping only.
> It is not real patient data and not clinical evidence.

**How to run (Cursor + Colab extension, A100):** select the Colab kernel with an A100 runtime, then **Run All**.
The notebook writes the whole project into `/content/hw3a` (the `%%writefile` cells in section 0) and runs every step.
Approximate run time: 45–75 min on an A100/L4, 1.5–2 h on a T4 (sizes are chosen automatically in section 0).
All figures and tables are displayed inline, so the saved notebook keeps the results. The last cell builds the report
and zips every output.

| Step | Section |
|---|---|
| Setup, source, unit tests | 0 |
| 1. Data inspection | 1 |
| 2. Baseline cDCGAN, plus a deliberately unstable run | 2 |
| 3. Evaluation protocol (judge classifier) | 3 |
| 4–5. Ray Tune smoke test + full ASHA search | 4 |
| 6. Select, stress-test, downstream comparison | 5 |
| 7–8. Ray Serve deploy, client, load test, autoscaling | 6 |
| Report, model card, reproducibility, packaging | 7 |
""")

md("## 0. Setup")
code("""
!nvidia-smi
import os, sys
PROJECT = "/content/hw3a" if os.path.exists("/content") else os.path.abspath("hw3a")
os.makedirs(PROJECT, exist_ok=True)
for d in ["gan_lab", "tests", "report"]:
    os.makedirs(os.path.join(PROJECT, d), exist_ok=True)
%cd {PROJECT}
print(sys.version)
""")
code("""
# Colab already has a CUDA build of torch/torchvision. Install the rest of the stack.
!pip install -q "ray[tune,serve]" medmnist tensorboardX markdown
import torch, ray
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("ray", ray.__version__, "| CPUs:", os.cpu_count())
""")
code("""
# Run settings chosen from the hardware. Big GPUs (A100/L4/H100) get the full study.
# Smaller ones (T4, V100) get a shorter but still complete search. Edit here to override.
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
NCPU = os.cpu_count() or 2
BIG = any(k in GPU for k in ("A100", "H100", "L4", "RTX", "5090", "4090", "3090"))
TUNE_SAMPLES = 24 if BIG else 16
TUNE_EPOCHS  = 40 if BIG else 30
BASE_EPOCHS  = 60 if BIG else 40
UNSTABLE_EPOCHS = 40 if BIG else 25
# Ray CPU accounting: Colab T4 machines have only 2 vCPUs, so request fractional CPUs.
# That way 4 tuning trials can share the GPU and Serve can still scale to 4 replicas.
CPT = max(1, NCPU // 4) if NCPU >= 8 else 0.5
SERVE_CPUS = 1 if NCPU >= 8 else 0.4
print(dict(GPU=GPU, CPUs=NCPU, TUNE_SAMPLES=TUNE_SAMPLES, TUNE_EPOCHS=TUNE_EPOCHS, BASE_EPOCHS=BASE_EPOCHS,
           CPT=CPT, SERVE_CPUS=SERVE_CPUS))
""")

md("### 0.1 Project source\nThese cells write the project files to the runtime. They are collapsed by default; expand them to read the code.")
for rel in SOURCES:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        src = f.read()
    code(f"%%writefile {rel}\n{src}", collapsed=True)

md("### 0.2 Install the package and run the unit tests (smoke tests)")
code("""
!pip install -q -e .
!python -m pytest -q
""")
code(SHOW)

md("## 1. Load and inspect PneumoniaMNIST")
code("""
!python inspect_data.py --out outputs/data
show("outputs/data/class_counts.png", "outputs/data/examples_per_class.png", "outputs/data/class_means_hist.png")
""")

md("""
## 3 (prerequisite). Judge classifier for the evaluation protocol
We train a frozen classifier on **real** data only. It provides the feature space for the Fréchet distance and diversity
metrics, and it checks class consistency of generated images. We train it before the GAN so the baseline run can log
these metrics.
""")
code("!python train_judge.py --out artifacts/judge.pt --epochs 25")

md("""
## 2. Baseline conditional DCGAN
Default config: lr 2e-4 for both networks, β1 = 0.5, z = 64, ngf = ndf = 64, one-sided label smoothing 0.1, 1 D step per G step.
Logged every epoch: D and G loss, D accuracy, D(x), D(G(z)), G gradient norm, and a fixed-noise grid. Every 5 epochs we
also log the evaluation metrics (TensorBoard event files are written to `runs/baseline/tb`).
""")
code("!python train_gan.py --epochs {BASE_EPOCHS} --eval-every 5 --out runs/baseline")
code("""
show("runs/baseline/curves.png")
for ep in sorted({1, 10, BASE_EPOCHS // 2, BASE_EPOCHS}):
    print(f"epoch {ep}  (top row: normal, bottom row: pneumonia; SYNTHETIC)")
    show(f"runs/baseline/grids/epoch_{ep:03d}.png", width=420)
""")

md("""
### 2b. A deliberately unstable configuration (diagnostics)
lr_D = 1e-3 (20× lr_G), 2 D steps per G step, no label smoothing. We expect the discriminator to overpower the generator:
D accuracy goes to 1, D(G(z)) goes to 0, and the generator stalls or collapses.
""")
code("!python train_gan.py --epochs {UNSTABLE_EPOCHS} --eval-every 5 --out runs/unstable --set lr_d=1e-3 lr_g=5e-5 label_smoothing=0.0 d_steps=2")
code("""
show("runs/unstable/curves.png")
show(f"runs/unstable/grids/epoch_{UNSTABLE_EPOCHS:03d}.png", width=420)
""")
code("""
# Optional: TensorBoard (renders in browser Colab; may not render inside Cursor). The PNG curves above carry the same data.
try:
    %load_ext tensorboard
    %tensorboard --logdir runs
except Exception as e:
    print("TensorBoard not available here:", e)
""")

md("""
## 4. Ray Tune: distributed hyper-parameter search
Search space: lr_G, lr_D ~ loguniform(1e-5, 5e-4); β1 ∈ {0, 0.5}; z ∈ {32, 64, 128}; width ∈ {32, 64}; label smoothing ∈ {0, 0.1};
D steps ∈ {1, 2}; spectral norm ∈ {F, T}; batch ∈ {64, 128}.
ASHA early-stops weak trials on `quality_score`. Each trial reserves 0.25 GPU, so 4 trials share the A100.
Checkpoints are reported every iteration.

### 4.1 Smoke test
""")
code("!python tune_gan.py --num-samples 2 --max-epochs 2 --smoke")
md("### 4.2 Full search")
code("""
print("samples:", TUNE_SAMPLES, "epochs:", TUNE_EPOCHS, "cpus per trial:", CPT)
!python tune_gan.py --num-samples {TUNE_SAMPLES} --max-epochs {TUNE_EPOCHS} --epochs-per-report 2 --gpus-per-trial 0.25 --cpus-per-trial {CPT}
""")
code("""
T = "outputs/tune_cgan_search"
print(json.dumps(json.load(open(f"{T}/tune_summary.json"))["best_config"], indent=1))
display(pd.read_csv(f"{T}/trials.csv").head(12))
show(f"{T}/trial_curves.png", f"{T}/hparam_scatter.png", f"{T}/resource_usage.png")
print("Best checkpoint grid (top: normal, bottom: pneumonia; SYNTHETIC)")
show(f"{T}/best_checkpoint_grid.png", width=420)
""")
code("""
# Ray resource log (stand-in for the dashboard screenshot, since port 8265 is not reachable from Cursor)
print(open(f"{T}/resource_log.csv").read()[:3000])
!ls ray_results/cgan_search | head -30
""")

md("""
## 5. Select and stress-test the best model, and run the downstream comparison
We generate fixed-seed grids, latent interpolations, and same-z class swaps, then run the nearest-neighbour memorisation
check and look at failure cases. The downstream comparison uses 5 seeds on the held-out test set:
real-only | class-weighted | oversample+augment | real+synthetic (0.5×, 1× gap) | synthetic-only (TSTR).
""")
code("!python evaluate.py --checkpoint artifacts/best_generator.pt --out outputs/eval --seeds 5 --clf-epochs 20")
code("""
E = "outputs/eval"
show(f"{E}/grid_normal_seed1234.png", f"{E}/grid_pneumonia_seed1235.png", width=520)
show(f"{E}/real_vs_synthetic.png", f"{E}/class_swap_same_z.png", f"{E}/latent_interpolation.png")
show(f"{E}/nn_distance_hist.png", f"{E}/memorization_nearest_neighbors.png", f"{E}/failure_class_inconsistent.png")
""")
code("""
display(Markdown(open(f"{E}/downstream_tables.md").read()))
show(f"{E}/downstream_bars.png", f"{E}/confusion_matrices.png")
s = json.load(open(f"{E}/eval_summary.json"))
print(json.dumps({"gan_metrics": s["gan_metrics"], "memorization": s["memorization"]}, indent=1))
""")
md("Comparison: the untuned baseline generator, evaluated with the same protocol (3 seeds).")
code("!python evaluate.py --checkpoint runs/baseline/generator.pt --out outputs/eval_baseline --seeds 3 --clf-epochs 20")

md("### Reproducible fixed grid (same checkpoint + seed → same hash)")
code("""
import re, subprocess
def run_grid(out):
    o = subprocess.run([sys.executable, "regenerate_grid.py", "--checkpoint", "artifacts/best_generator.pt",
                        "--seed", "1234", "--out", out], capture_output=True, text=True).stdout
    print(o)
    return re.search(r"images sha256=(\\w+)", o).group(1)
h1, h2 = run_grid("outputs/fixed_grid_run1.png"), run_grid("outputs/fixed_grid_run2.png")
json.dump({"hash_run1": h1, "hash_run2": h2, "identical": h1 == h2,
           "command": "python regenerate_grid.py --checkpoint artifacts/best_generator.pt --seed 1234"},
          open("outputs/repro.json", "w"), indent=1)
print("identical:", h1 == h2)
show("outputs/fixed_grid_run1.png", width=520)
""")

md("""
## 6. Ray Serve deployment
The deployment uses `@serve.deployment` with autoscaling (1–4 replicas, target 4 ongoing requests per replica),
0.25 GPU and 1 CPU per replica, `@serve.batch`, a FastAPI ingress with pydantic validation, and responses marked as synthetic.
""")
code("""
import ray
from ray import serve
ray.shutdown()
ray.init(runtime_env={"env_vars": {"PYTHONPATH": PROJECT}}, ignore_reinit_error=True)
sys.path.insert(0, PROJECT)
from serve_app import deploy, replica_count
deploy("artifacts/best_generator.pt", num_gpus=0.25, max_replicas=4, num_cpus=SERVE_CPUS)
print("replicas:", replica_count())
""")
code("""
import requests
print(requests.get("http://127.0.0.1:8000/health").json())
print(json.dumps(requests.get("http://127.0.0.1:8000/model_card").json(), indent=1)[:1500])
!python client.py --url http://127.0.0.1:8000 --out outputs/serve
show("outputs/serve/api_grid_label0.png", "outputs/serve/api_grid_label1.png", width=420)
""")
md("### Load test + autoscaling trace")
code("""
from load_test import run_load_test
df, trace = run_load_test("http://127.0.0.1:8000", levels=(1, 4, 16, 32, 64), duration_s=20, n_images=8,
                          probe=replica_count, out="outputs/serve")
display(df)
show("outputs/serve/load_test.png")
""")
code("""
# Text evidence from Ray (the dashboard at :8265 is not reachable from Cursor)
!serve status
!ray status
!ray list actors --limit 30
""")
code("serve.shutdown()\nray.shutdown()")

md("## 7. Report, model card, and packaging")
code("""
!python make_report.py --out outputs/report
display(Markdown(open("outputs/report/REPORT.md").read().replace("](../../", "](")))
""")
code("""
import shutil
!cp report/MODEL_CARD.md outputs/report/
# Bundle results (small files only; full Ray trial dirs stay on the runtime)
!zip -qr hw3a_results.zip outputs artifacts/best_generator.pt artifacts/judge.pt artifacts/judge_metrics.json runs/baseline/history.json runs/baseline/curves.png runs/baseline/config.json runs/unstable/history.json runs/unstable/curves.png runs/baseline/tb runs/unstable/tb
!ls -lh hw3a_results.zip
try:
    from google.colab import files
    files.download("hw3a_results.zip")
except Exception as e:
    print("Automatic download unavailable here (", e, "). The executed notebook keeps all figures inline.")
""")
code("""
# Print the key numbers so they are saved in this notebook's outputs
summary = {
    "data": json.load(open("outputs/data/data_summary.json"))["splits"],
    "tune": json.load(open("outputs/tune_cgan_search/tune_summary.json")),
    "eval": {k: v for k, v in json.load(open("outputs/eval/eval_summary.json")).items() if k != "downstream"},
    "downstream_means": json.load(open("outputs/eval/eval_summary.json")).get("downstream", {}).get("means"),
    "load_test": pd.read_csv("outputs/serve/load_test.csv").to_dict(orient="records"),
    "repro": json.load(open("outputs/repro.json")),
}
print(json.dumps(summary, indent=1, default=str))
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:03d}"
out = os.path.join(ROOT, "HW3a_ConditionalGAN.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, "with", len(cells), "cells")
