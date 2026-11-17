# HW3a: Conditional GAN for synthetic PneumoniaMNIST (Ray Tune + Ray Serve)

> All generated images are **SYNTHETIC**. They are for education and research prototyping only. They are not patient data and not clinical evidence.

## Quick start on a local NVIDIA GPU (RTX 5090 lab machine)
The RTX 5090 is a Blackwell GPU (compute capability sm_120), so it needs a **CUDA 12.8+ build of PyTorch**.
The default `pip install torch` from PyPI may be an older build that fails with "no kernel image".
Linux or WSL2 is recommended for Ray. Native Windows works for Ray Tune, but Ray lists Windows support as beta.
```bash
git clone https://github.com/NAMAN-CHHEDA/Conditional-GANs-for-Responsible-Synthetic-Medical-Imaging.git
cd Conditional-GANs-for-Responsible-Synthetic-Medical-Imaging
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt && pip install -e .
python run_pipeline.py --check      # confirms the 5090 + torch build work (runs a real CUDA kernel)
python run_pipeline.py --smoke      # short end-to-end sanity run
python run_pipeline.py              # full study; logs in outputs/logs/, report in outputs/report/
```
`run_pipeline.py` runs every stage (tests, data, judge, baseline, unstable GAN, Ray Tune, evaluation, reproducibility check,
Ray Serve load test, report). `--only` and `--skip` select stages. On a 5090 the profile runs **5 Ray Tune trials at once**
(0.2 GPU each, 32 trials, 40 epochs), because these GANs are tiny and one trial cannot saturate the card.
Per-stage timings are saved to `outputs/timings.json`. The Ray dashboard is at http://127.0.0.1:8265 while Tune/Serve run, so
you can take the dashboard screenshots the assignment asks for.

## Quick start (Cursor + Colab extension)
1. Open `HW3a_ConditionalGAN.ipynb` and select the Colab kernel with an **A100** runtime.
2. **Run All.** The notebook writes the whole project to `/content/hw3a`, so no upload is needed, and then runs every step:
   unit tests → data inspection → judge → baseline + unstable GAN → Ray Tune smoke + full search → evaluation +
   downstream comparison → Ray Serve deployment, client and load test → report and zip. This takes about 45–75 minutes.
3. **Save the notebook** (Ctrl+S). Its outputs keep every figure and table. The last cell also tries to download `hw3a_results.zip`.
4. Turn `outputs/report/REPORT.html` into a PDF (open it in a browser and print to PDF) for the 6–10 page report.

If you edit any `.py` file, run `python build_notebook.py` to regenerate the notebook.

## Local / Linux usage
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install torch torchvision                          # pick your CUDA build: pytorch.org/get-started/locally
pip install -r requirements.txt && pip install -e .
python -m pytest -q                                    # 25 unit tests, CPU, ~15 s
python inspect_data.py                                 # Step 1
python train_judge.py                                  # frozen real-data judge for the metrics
python train_gan.py --epochs 60 --out runs/baseline    # Step 2 (TensorBoard: runs/baseline/tb)
python train_gan.py --epochs 40 --out runs/unstable --set lr_d=1e-3 lr_g=5e-5 label_smoothing=0.0 d_steps=2
python tune_gan.py --num-samples 2 --max-epochs 2 --smoke                       # Step 5 smoke
python tune_gan.py --num-samples 24 --max-epochs 40 --gpus-per-trial 0.25       # Step 5 full
python evaluate.py --checkpoint artifacts/best_generator.pt --seeds 5           # Steps 3 + 6
python regenerate_grid.py --checkpoint artifacts/best_generator.pt --seed 1234  # fixed grid + hash
CHECKPOINT=artifacts/best_generator.pt serve run serve_app:app                  # Step 7
python client.py && python load_test.py --probe                                 # Step 8
python make_report.py                                                           # report (md + html)
```
On Windows, Ray works for local smoke tests (Ray lists Windows support as beta). Use WSL2, Linux or Colab for the full runs.

## Layout
| Path | Purpose |
|---|---|
| `gan_lab/data.py` | MedMNIST loading, [-1,1] GAN space, 28↔64 round trip (shortcut control), class statistics |
| `gan_lab/models.py` | Conditional generator (label embedding ⊕ z), projection discriminator (+ optional spectral norm), SmallCNN |
| `gan_lab/trainer.py` | Adversarial loop (non-saturating BCE, one-sided smoothing, D-step ratio, full instrumentation) |
| `gan_lab/metrics.py` | Judge-feature Fréchet distance, diversity / collapse, NN memorisation vs held-out real, quality score |
| `gan_lab/classifier.py` | Downstream classifier, naive augmentation, AUC / F1 / balanced accuracy / confusion matrix |
| `gan_lab/service.py` | Validated request model, batched generation, PNG provenance metadata |
| `tune_gan.py` | Ray Tune: search space, ASHA, fractional-GPU trials, checkpoints, resume, analysis, best-model export |
| `evaluate.py` | Grids, interpolation, class swap, memorisation, failure cases, 5-seed downstream study with paired stats |
| `serve_app.py` | Ray Serve deployment (FastAPI ingress, autoscaling, `@serve.batch`, `/generate`, `/health`, `/model_card`) |
| `client.py`, `load_test.py` | Example client and validation checks; concurrency sweep + replica trace |
| `make_report.py` | Builds `outputs/report/REPORT.md` and a self-contained `REPORT.html` from the results |
| `report/MODEL_CARD.md` | One-page model card |

## Rubric map
| Criterion | Where |
|---|---|
| Data understanding (5) | `inspect_data.py`, report §2 (splits, label mapping from MedMNIST INFO, [-1,1] check, imbalance, prior shift) |
| GAN implementation (9) | `models.py`, `trainer.py`, report §3–4 (projection conditioning, detach, non-saturating loss, instrumentation, unstable run) |
| Diagnostics & evaluation (9) | `metrics.py`, `evaluate.py`, report §6–7 (FD, diversity, memorisation, failure cases, 5-seed paired downstream vs strong baselines, TSTR) |
| Ray Tune (8) | `tune_gan.py`, report §5 (search space, ASHA, resources, checkpoints, resume, CSV analysis, seeded search) |
| Ray Serve (7) | `serve_app.py`, `client.py`, `load_test.py`, report §8 |
| Ethics (5) | report §10, `MODEL_CARD.md`, API safety fields |
| Reproducibility (4) | seeds, `regenerate_grid.py` hash check, `pytest`, this README |
| Presentation (3) | `make_report.py` → REPORT.html |
