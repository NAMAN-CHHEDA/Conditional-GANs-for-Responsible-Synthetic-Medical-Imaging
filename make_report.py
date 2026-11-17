"""Assemble the technical report (Markdown + self-contained HTML) from the experiment
outputs. Numbers and verdict sentences are filled in from the JSON/CSV results so the
report always matches the run that produced it.

python make_report.py --out outputs/report
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))


def jload(path, default=None):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def dig(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def md_table(rows, cols, headers=None):
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def fig(path, caption, width=None):
    if not os.path.exists(os.path.join(ROOT, path)):
        return f"*(figure missing: `{path}`)*"
    w = f'{{width="{width}"}}' if width else ""
    return f"![{caption}](../../{path}){w}\n\n*{caption}*"


def build(out_dir):
    data = jload("outputs/data/data_summary.json", {})
    judge = jload("artifacts/judge_metrics.json", {})
    base_hist = jload("runs/baseline/history.json", [])
    unst_hist = jload("runs/unstable/history.json", [])
    smoke = jload("outputs/tune_cgan_smoke/tune_summary.json", {})
    tune = jload("outputs/tune_cgan_search/tune_summary.json", {})
    ev = jload("outputs/eval/eval_summary.json", {})
    ev_base = jload("outputs/eval_baseline/eval_summary.json", {})
    client = jload("outputs/serve/client_log.json", {})
    repro = jload("outputs/repro.json", {})
    trials_csv = os.path.join(ROOT, "outputs/tune_cgan_search/trials.csv")
    trials = pd.read_csv(trials_csv) if os.path.exists(trials_csv) else None
    lt_csv = os.path.join(ROOT, "outputs/serve/load_test.csv")
    lt = pd.read_csv(lt_csv) if os.path.exists(lt_csv) else None
    rt_csv = os.path.join(ROOT, "outputs/serve/replica_trace.csv")
    rtrace = pd.read_csv(rt_csv) if os.path.exists(rt_csv) else None

    S = []  # report sections
    tr = dig(data, "splits", "train", default={})
    gm = ev.get("gan_metrics", {})
    mem = ev.get("memorization", {})
    ds = ev.get("downstream", {})
    means = ds.get("means", {})
    paired = pd.DataFrame(ds.get("paired", []))

    def delta(cond, metric):
        if paired.empty:
            return None
        r = paired[(paired.condition == cond) & (paired.metric == metric)]
        return None if r.empty else r.iloc[0].to_dict()

    d_syn = delta("real_plus_synth_1x", "f1_macro")
    d_cw = delta("real_class_weighted", "f1_macro")
    d_os = delta("real_oversample_aug", "f1_macro")
    d_syn_auc = delta("real_plus_synth_1x", "auc")

    def verdict(d):
        if d is None:
            return "could not be assessed (missing results)"
        lo, hi = d["ci95_low"], d["ci95_high"]
        s = f"{d['mean_delta']:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}, paired t p={d['p_value_paired_t']:.3f}, wins {d['wins']})"
        if lo > 0:
            return s + " - a statistically reliable improvement"
        if hi < 0:
            return s + " - a statistically reliable degradation"
        return s + " - not distinguishable from zero with this many seeds"

    # ------------------------------------------------------------------ title + summary
    S.append("# Conditional GANs for Synthetic PneumoniaMNIST: Ray Tune search, task-grounded evaluation, and Ray Serve deployment\n")
    S.append("**DATA 255 - Homework 3a.** All generated images in this report are **SYNTHETIC**. They are "
             "not patient data and not clinical evidence. They are for education and research prototyping only.\n")
    S.append("## 1. Summary\n")
    S.append(
        "**Question.** Can a class-conditional GAN make normal and pneumonia images that are diverse enough "
        "to help a small downstream classifier under class imbalance? And how do we tell when the synthetic "
        "images are misleading?\n\n"
        f"**What we built.** We built a conditional DCGAN with a projection discriminator at 64x64. We tuned it with "
        f"Ray Tune: {fmt(tune.get('num_trials'))} trials, ASHA early stopping, and fractional-GPU trials. "
        f"We selected the model with a composite score: validation F1 gain minus collapse and memorisation penalties. "
        f"We evaluated it with qualitative grids, a domain-specific Frechet distance, diversity and memorisation checks, "
        f"and a {len(ds.get('seeds', []))}-seed downstream comparison on the held-out test set. Finally, we deployed it as "
        f"an autoscaling Ray Serve API.\n\n"
        f"**Answer.** Adding synthetic minority-class (normal) images to the training set changed test macro-F1 by "
        f"{verdict(d_syn)}, compared with training on real data only. "
        f"The simple baselines that need no generator changed it by {verdict(d_cw)} (class-weighted loss) and "
        f"{verdict(d_os)} (oversampling plus naive augmentation). The fair test for the GAN is therefore whether it "
        "beats these cheap baselines, not whether it beats doing nothing.\n")

    # ------------------------------------------------------------------ data
    S.append("## 2. Data understanding and preprocessing\n")
    rows = []
    for s in ("train", "val", "test"):
        info = dig(data, "splits", s, default={})
        c = info.get("counts", {})
        rows.append({"split": s, "n": info.get("n"), "normal (0)": c.get("normal"), "pneumonia (1)": c.get("pneumonia"),
                     "normal fraction": info.get("fraction_normal")})
    S.append(md_table(rows, ["split", "n", "normal (0)", "pneumonia (1)", "normal fraction"]))
    S.append(
        f"\n\nThe label mapping comes from the MedMNIST metadata: `{data.get('label_mapping')}`. The minority class "
        f"is **normal**. Train has a pneumonia:normal ratio of {fmt(tr.get('imbalance_ratio_pneumonia_to_normal'), 2)}:1. "
        "The test split is noticeably less imbalanced (37.5% normal vs 25.8%). This is a prior shift, so a "
        "classifier tuned to the training prior will under-call normal on test. That is one reason we report "
        "balanced accuracy, macro-F1, and AUC rather than accuracy.\n\n"
        "Preprocessing: the 28x28 uint8 images are scaled to [0,1], bilinearly resized to 64x64 (as specified in the "
        "assignment), and normalised to [-1,1], which matches the generator's Tanh output. "
        f"The assignment's torchvision `Resize->ToTensor->Normalize` pipeline gives shape "
        f"{dig(data, 'torchvision_check', 'sample_shape')} with range "
        f"[{fmt(dig(data, 'torchvision_check', 'min'), 2)}, {fmt(dig(data, 'torchvision_check', 'max'), 2)}]. "
        "Our in-memory tensor pipeline matches it to within a mean absolute difference of "
        f"{fmt(dig(data, 'torchvision_check', 'mean_abs_diff_vs_tensor_pipeline'), 4)}.\n\n"
        "*Shortcut control.* The generator outputs 64x64 images, which we downsample back to 28x28 for every "
        "classifier and memorisation check. We pass all real images through the same 28->64->28 round trip. "
        "Otherwise a classifier could learn \"slightly blurrier means synthetic, which means normal\" "
        "instead of anatomy.\n")
    S.append(fig("outputs/data/class_counts.png", "Figure 1. Class counts per split."))
    S.append(fig("outputs/data/examples_per_class.png", "Figure 2. 16 real training images per class."))
    S.append(fig("outputs/data/class_means_hist.png", "Figure 3. Per-class mean images and intensity histograms."))

    # ------------------------------------------------------------------ method
    S.append("## 3. Method\n")
    S.append(
        "**Objective.** A GAN is a two-player minimax game, "
        "min_G max_D E[log D(x|y)] + E[log(1 - D(G(z|y)|y))]. "
        "D is a binary classifier between real and fake images, given the label. G maps noise z and label y to an image. "
        "We use BCE-with-logits and the *non-saturating* generator loss (G maximises log D(G(z|y)|y)). "
        "With the original loss, G's gradients vanish exactly when D is winning. "
        "Real targets use one-sided label smoothing (1 - s) so that D does not become overconfident. During the D update, fake "
        "images are `detach()`-ed so that no gradient flows into G.\n\n"
        "**Generator G(z, y).** The label goes through a 32-d `nn.Embedding` and is concatenated with z. Then come ConvTranspose "
        "blocks 1->4->8->16->32->64 with BatchNorm+ReLU and a Tanh output. Weights use DCGAN N(0, 0.02) initialisation.\n\n"
        "**Discriminator D(x, y).** It uses strided convolutions 64->32->16->8->4 with LeakyReLU(0.2), then global sum pooling to h, and "
        "outputs a *projection* logit w^T h + embed(y)^T h (Miyato & Koyama, 2018). Projection gives D a class-specific "
        "direction in feature space, so it can penalise \"a realistic image of the wrong class\". This is stronger than "
        "concatenating a label map at the input. Spectral normalisation is optional and searched by Tune; when it is on, "
        "BatchNorm in D is removed.\n\n"
        "**Instrumentation.** Every epoch we log D and G loss, D accuracy on real and fake, mean D(x) and D(G(z)), and G "
        "gradient norm, plus a fixed-noise grid (TensorBoard + PNG). Every few epochs we also compute the evaluation "
        "metrics below.\n\n"
        "**Evaluation protocol (beyond visual quality).** A frozen *judge* CNN is trained on real data only, with "
        f"class-weighted loss (val AUC {fmt(dig(judge, 'val', 'auc'))}, test AUC {fmt(dig(judge, 'test', 'auc'))}). "
        "It provides:\n"
        "1. *Class consistency*: the fraction of images generated for class c that the judge labels c.\n"
        "2. *Frechet distance* between judge features of real and fake images, per class. This is an FID-style score with a domain-relevant "
        "feature extractor; ImageNet Inception features are not meaningful for 28x28 chest X-rays.\n"
        "3. *Diversity ratio*: mean pairwise feature distance of fakes divided by that of reals. A value near 1 means real-like spread; "
        "a value well below 1 means mode collapse. collapse_penalty = mean_c max(0, 1 - ratio_c).\n"
        "4. *Memorisation*: nearest-neighbour pixel RMSE from each fake to the training set, compared with "
        "the same distance for **held-out real validation images**. A fake is flagged if it is closer to a training "
        "image than 95% of real validation images are. Held-out data is flagged 5% of the time by construction, so "
        "memorization_penalty = excess flag rate over 5%.\n"
        "5. *Downstream utility*: macro-F1 on validation of a quick classifier trained on real + synthetic-minority data, "
        "minus the same classifier trained on real data + the same number of **duplicated real minority images**.\n\n"
        "The quality score is `quality_score = val_f1_gain - 0.5 * collapse_penalty - 0.5 * memorization_penalty`.\n\n"
        "*Why duplicated minority images as the control?* During the smoke test, a generator one epoch old, producing "
        "near-noise images, still gained +0.12 validation F1 over a real-only baseline. Adding *any* minority-labelled "
        "samples shifts the class prior and helps macro-F1 on an imbalanced split. Comparing against random oversampling "
        "isolates the contribution of the synthetic image *content*. This is an example of the misleading evaluation the assignment "
        "warns about.\n")

    # ------------------------------------------------------------------ baseline training
    S.append("## 4. Baseline training dynamics and failure modes\n")
    if base_hist:
        last = base_hist[-1]
        evals = [h for h in base_hist if "quality_score" in h]
        le = evals[-1] if evals else {}
        S.append(
            f"The baseline (default config, {last['epoch']} epochs) ended with D loss {fmt(last['d_loss'])}, "
            f"G loss {fmt(last['g_loss'])}, D accuracy {fmt(last['d_acc'])}, mean D(x) {fmt(last['d_real'])}, and mean D(G(z)) "
            f"{fmt(last['d_fake'])}. At the final evaluation: Frechet distance {fmt(le.get('fd_mean'))}, diversity ratio "
            f"{fmt(le.get('div_ratio_0'))}/{fmt(le.get('div_ratio_1'))} (normal/pneumonia), class consistency "
            f"{fmt(le.get('class_consistency'))}, memorisation flag rate {fmt(le.get('mem_rate'))}.\n")
    if unst_hist:
        ul = unst_hist[-1]
        ue = [h for h in unst_hist if "quality_score" in h]
        S.append(
            f"For contrast, we ran a deliberately **unbalanced** configuration (lr_D = 1e-3, 20x lr_G; two D steps per G step; no "
            f"label smoothing). It shows the textbook *discriminator overpowering* pattern: D accuracy {fmt(ul['d_acc'])}, "
            f"mean D(G(z)) {fmt(ul['d_fake'])}, G loss {fmt(ul['g_loss'])}, and G gradient norm {fmt(ul['g_grad_norm'])}. "
            f"Its final Frechet distance was {fmt(ue[-1].get('fd_mean') if ue else None)} and its diversity ratio "
            f"{fmt(ue[-1].get('div_ratio_0') if ue else None)}/{fmt(ue[-1].get('div_ratio_1') if ue else None)}. "
            "When D accuracy sits near 1 and D(G(z)) near 0, G gets almost no useful signal, and samples freeze or "
            "collapse. Healthy runs keep D accuracy around 0.6-0.85 and D(x) and D(G(z)) apart, but not saturated.\n")
    S.append(fig("runs/baseline/curves.png", "Figure 4. Baseline training curves (losses, D balance, metrics)."))
    S.append(fig("runs/unstable/curves.png", "Figure 5. Unbalanced configuration: the discriminator overpowers the generator."))

    # ------------------------------------------------------------------ ray tune
    S.append("## 5. Ray Tune hyper-parameter search\n")
    S.append(
        "| hyper-parameter | space | why it matters |\n|---|---|---|\n"
        "| lr_G, lr_D | loguniform(1e-5, 5e-4) each | the D/G learning-rate balance is the main lever on stability |\n"
        "| beta1 | {0.0, 0.5} | low momentum (DCGAN's 0.5, or 0) reduces oscillation in adversarial games |\n"
        "| latent_dim | {32, 64, 128} | capacity for intra-class variation (diversity vs. difficulty) |\n"
        "| ngf = ndf | {32, 64} | network width / capacity balance |\n"
        "| label_smoothing | {0.0, 0.1} | prevents an overconfident D |\n"
        "| d_steps | {1, 2} | discriminator update ratio |\n"
        "| spectral_norm (D) | {False, True} | Lipschitz control, an alternative to BatchNorm in D |\n"
        "| batch_size | {64, 128} | gradient noise and BatchNorm statistics |\n")
    asha = tune.get("asha", {})
    S.append(
        f"\n**Execution.** Random search ({fmt(tune.get('num_trials'))} trials, seeded `BasicVariantGenerator`) with "
        f"**ASHA** (`max_t`={asha.get('max_t')} iterations of {asha.get('epochs_per_iteration')} epochs, grace period "
        f"{asha.get('grace_period')}, reduction factor 2) on `quality_score`. Each trial reserves "
        f"`{tune.get('resources_per_trial')}`, so four trials share one A100 and Ray's resource accounting prevents GPU "
        "oversubscription. The dataset and judge weights are put in the object store once (`tune.with_parameters`). "
        "Every iteration reports a checkpoint (G, D, both optimisers, epoch, sample grid). `CheckpointConfig` keeps the 2 best "
        "per trial by quality_score, and `get_checkpoint()` makes trials resumable (`FailureConfig(max_failures=1)`). "
        f"Smoke test: {fmt(smoke.get('num_trials'))} trials in {fmt(smoke.get('wall_clock_min'), 1)} min. "
        f"Full search: {fmt(tune.get('wall_clock_min'), 1)} min wall-clock, with "
        f"{fmt(tune.get('num_stopped_early'))}/{fmt(tune.get('num_trials'))} trials stopped early by ASHA and "
        f"{fmt(tune.get('num_errors'))} errors.\n")
    bc = tune.get("best_config", {})
    bm = tune.get("best_metrics", {})
    S.append(
        f"**Best configuration** (selected epoch {fmt(tune.get('best_selected_epoch'))}): "
        + ", ".join(f"`{k}={fmt(bc.get(k), 6) if isinstance(bc.get(k), float) else bc.get(k)}`"
                    for k in ["lr_g", "lr_d", "beta1", "latent_dim", "ngf", "label_smoothing", "d_steps",
                              "spectral_norm", "batch_size"])
        + f". At selection: quality_score {fmt(bm.get('quality_score'))}, val F1 gain {fmt(bm.get('val_f1_gain'))}, "
          f"collapse penalty {fmt(bm.get('collapse_penalty'))}, memorisation penalty {fmt(bm.get('memorization_penalty'))}, "
          f"FD {fmt(bm.get('fd_mean'))}, class consistency {fmt(bm.get('class_consistency'))}.\n")
    if trials is not None and len(trials):
        top = trials.head(6).to_dict(orient="records")
        S.append("\nTop trials by best quality_score:\n\n" + md_table(
            top, ["lr_g", "lr_d", "beta1", "latent_dim", "ngf", "d_steps", "spectral_norm", "epochs",
                  "best_quality_score", "val_f1_gain@best", "collapse_penalty@best", "fd_mean@best"],
            ["lr_G", "lr_D", "b1", "z", "ngf", "d_steps", "SN", "epochs", "Q", "dF1", "collapse", "FD"]) + "\n")
        sn = trials.groupby("spectral_norm")["best_quality_score"].median().to_dict()
        S.append(f"\nMedian best quality_score by spectral norm: {', '.join(f'{k}: {v:.3f}' for k, v in sn.items())}. "
                 f"Median by d_steps: {', '.join(f'{k}: {v:.3f}' for k, v in trials.groupby('d_steps')['best_quality_score'].median().items())}.\n")
    S.append(fig("outputs/tune_cgan_search/trial_curves.png",
                 "Figure 6. quality_score, Frechet distance, and D accuracy per trial (best trial bold)."))
    S.append(fig("outputs/tune_cgan_search/hparam_scatter.png", "Figure 7. Best score vs each hyper-parameter."))
    S.append(fig("outputs/tune_cgan_search/resource_usage.png", "Figure 8. Ray CPU/GPU reservations during the search."))

    # ------------------------------------------------------------------ evaluation
    S.append("## 6. Evaluating and stress-testing the selected generator\n")
    S.append(md_table([
        {"metric": "class consistency (normal / pneumonia)", "value": f"{fmt(gm.get('class_consistency_0'))} / {fmt(gm.get('class_consistency_1'))}"},
        {"metric": "Frechet distance, judge features (normal / pneumonia)", "value": f"{fmt(gm.get('fd_0'))} / {fmt(gm.get('fd_1'))}"},
        {"metric": "feature diversity ratio (normal / pneumonia)", "value": f"{fmt(gm.get('div_ratio_0'))} / {fmt(gm.get('div_ratio_1'))}"},
        {"metric": "pixel diversity ratio (normal / pneumonia)", "value": f"{fmt(gm.get('pix_div_ratio_0'))} / {fmt(gm.get('pix_div_ratio_1'))}"},
        {"metric": "collapse penalty", "value": fmt(gm.get("collapse_penalty"))},
        {"metric": "memorisation flag rate (held-out real = 0.05)", "value": fmt(mem.get("flag_rate"))},
        {"metric": "median NN distance fake / real-val", "value": fmt(gm.get("nn_ratio"))},
        {"metric": "near-exact duplicates of training images (RMSE<0.001)", "value": fmt(mem.get("n_near_exact_duplicates"))},
        {"metric": "class-inconsistent samples out of 1000 per class", "value": str(mem.get("n_class_inconsistent"))},
    ], ["metric", "value"]))
    if ev_base.get("gan_metrics"):
        b = ev_base["gan_metrics"]
        S.append(f"\n\nFor comparison, the untuned baseline generator has FD {fmt(b.get('fd_mean'))}, diversity "
                 f"{fmt(b.get('div_ratio_0'))}/{fmt(b.get('div_ratio_1'))}, consistency {fmt(b.get('class_consistency'))}, "
                 f"and memorisation flag rate {fmt(b.get('mem_rate'))}.")
    S.append(
        "\n\n**Reading the memorisation check.** A flag rate near 0.05 and an NN ratio near or above 1 mean the generator is "
        "no closer to its training images than unseen real X-rays are. That is evidence against copying, but it is "
        "*not* a privacy guarantee. Pixel RMSE misses shifted or contrast-changed copies, and membership-inference "
        "attacks can succeed without near-duplicates. A flag rate well above 0.05 would be a red flag. The closest "
        "pairs are shown in Figure 12 for manual inspection.\n")
    S.append(fig(f"outputs/eval/grid_normal_seed1234.png", "Figure 9a. SYNTHETIC normal, fixed seed 1234 (64 samples)."))
    S.append(fig(f"outputs/eval/grid_pneumonia_seed1235.png", "Figure 9b. SYNTHETIC pneumonia, fixed seed 1235 (64 samples)."))
    S.append(fig("outputs/eval/real_vs_synthetic.png", "Figure 10. Real vs synthetic at the classifier's 28x28 resolution."))
    S.append(fig("outputs/eval/class_swap_same_z.png",
                 "Figure 11. Same latent z, label normal (top) vs pneumonia (bottom): what conditioning changes."))
    S.append(fig("outputs/eval/latent_interpolation.png", "Figure 11b. Latent interpolations (smooth = no memorised lookup table)."))
    S.append(fig("outputs/eval/memorization_nearest_neighbors.png", "Figure 12. Closest synthetic samples and their 3 nearest training images."))
    S.append(fig("outputs/eval/nn_distance_hist.png", "Figure 13. NN distance to train: synthetic vs held-out real."))
    S.append(fig("outputs/eval/failure_class_inconsistent.png", "Figure 14. Failure cases: the judge confidently assigns the other class."))
    S.append(
        "\n**Failure analysis.** (i) *Class leakage*: some images generated as normal carry diffuse opacities that "
        "the judge reads as pneumonia, and the reverse also happens (Figure 14). Labels on synthetic images are therefore *noisy*, which directly "
        "limits augmentation. (ii) *Low-frequency bias*: at 28x28, the GAN reproduces global shape, contrast, and lung fields well, "
        "but fine texture is smoothed. A classifier that relies on texture gains little. (iii) *Mode coverage*: diversity "
        "ratios below 1 show that the generator under-represents the tails, such as unusual patient positioning, devices, and "
        "very small children. Synthetic data cannot add variation the GAN never learned. (iv) *Metric gaming*: "
        "the composite score can be raised by shifting class priors (Section 3) or by blurry average images that lower "
        "pixel-NN distance. This is why we report the score's components separately and inspect the grids.\n")

    # ------------------------------------------------------------------ downstream
    S.append("## 7. Downstream classifier: does synthetic data help?\n")
    S.append(
        f"Protocol: the same SmallCNN, {ds.get('epochs')} epochs, Adam. Model selection uses the best validation macro-F1 epoch. "
        f"We use {len(ds.get('seeds', []))} seeds, and each seed fixes the initialisation, data order, and synthetic sample. All conditions are scored on "
        "the **same held-out test set**, which is never used for tuning. Conditions: real-only; class-weighted loss; "
        "random minority oversampling + naive augmentation (small rotation, translation, and intensity changes; no flips, because "
        "flips would put the heart on the wrong side); real + synthetic normals filling 50% or 100% of the class gap; and "
        "synthetic-only training (train-on-synthetic, test-on-real, TSTR), which measures conditional fidelity.\n")
    tbl = os.path.join(ROOT, "outputs/eval/downstream_tables.md")
    if os.path.exists(tbl):
        S.append(open(tbl, encoding="utf-8").read())
    S.append(fig("outputs/eval/downstream_bars.png", "Figure 15. Test metrics per condition (mean ± sd over seeds)."))
    S.append(fig("outputs/eval/confusion_matrices.png", "Figure 16. Test confusion matrices summed over seeds."))
    tstr = means.get("synthetic_only_TSTR", {})
    ro = means.get("real_only", {})
    S.append(
        f"\n**Interpretation.** Real + synthetic (1x gap) vs real-only, macro-F1: {verdict(d_syn)}. AUC: {verdict(d_syn_auc)}. "
        f"Class weighting: {verdict(d_cw)}. Oversampling + augmentation: {verdict(d_os)}. "
        f"A classifier trained *only* on synthetic images reaches test AUC {fmt(tstr.get('auc'))}, compared with "
        f"{fmt(ro.get('auc'))} for real-only. The GAN therefore captures much of the class-discriminative signal, but not all of it. "
        "Most of the macro-F1 movement in every 'rebalancing' condition comes from the decision threshold "
        "shifting toward the minority class; AUC, which does not depend on the threshold, moves much less. The honest conclusion is "
        "that synthetic augmentation should be judged against class weighting and oversampling, and that its benefit, "
        "if any, is small relative to seed-to-seed variance. The test-set prior shift (37.5% normal) also "
        "inflates the apparent value of any method that predicts 'normal' more often.\n")

    # ------------------------------------------------------------------ serve
    S.append("## 8. Ray Serve deployment\n")
    S.append(
        "**Endpoint contract.** `POST /generate` with JSON `{label: 0|1|\"normal\"|\"pneumonia\", n: 1..64, "
        "seed?: int >= 0, size: 28|64, return_grid?: bool}`. The response carries `synthetic: true`, a disclaimer, "
        "`label_name`, the seed used (server-chosen if omitted, so results are reproducible), base64 PNGs, the "
        "model id (checkpoint SHA-256 prefix) and selected epoch, the replica id, and latency. Every PNG embeds tEXt metadata "
        "(`Synthetic=true`, generator id, label, seed), and the HTTP response carries an `X-Synthetic-Data: true` header. "
        "`GET /health` and `GET /model_card` are also provided.\n\n"
        "**Deployment.** `@serve.deployment` with `autoscaling_config` (1-4 replicas, target 4 ongoing "
        "requests per replica, upscale delay 3 s, downscale delay 30 s), `ray_actor_options={num_cpus: 1, num_gpus: 0.25}`, "
        "`max_ongoing_requests=16`, and a FastAPI ingress. `@serve.batch(max_batch_size=16, batch_wait_timeout_s=0.005)` "
        "merges concurrent requests into one generator forward pass. Because G runs in eval mode, BatchNorm uses "
        "running statistics, so a request's images do not depend on which other requests share its batch. "
        "We unit-tested this property.\n")
    g0 = client.get("generate_label0", {})
    if g0:
        S.append("Example request and (truncated) response:\n\n```json\n" + json.dumps(g0.get("request"), indent=1)
                 + "\n" + json.dumps(g0.get("response"), indent=1)[:1500] + "\n```\n")
    if client.get("validation"):
        S.append("Input validation (pydantic, `extra=\"forbid\"`):\n\n" + md_table(
            [{"request": json.dumps(v["request"]), "status": v["status"], "message": v["msg"]} for v in client["validation"]],
            ["request", "status", "message"]) + f"\n\nSame seed gives identical images: **{client.get('seed_reproducible')}**.\n")
    S.append(fig("outputs/serve/api_grid_label0.png", "Figure 17a. Grid returned by the API (normal, seed 42), SYNTHETIC."))
    S.append(fig("outputs/serve/api_grid_label1.png", "Figure 17b. Grid returned by the API (pneumonia, seed 42), SYNTHETIC."))
    if lt is not None:
        S.append("\n**Load test** (threads posting n=8 images per request, 20 s per concurrency level):\n\n" + md_table(
            lt.to_dict(orient="records"),
            ["concurrency", "requests", "errors", "throughput_rps", "images_per_s", "p50_ms", "p95_ms", "p99_ms"]) + "\n")
        peak = lt.loc[lt["throughput_rps"].idxmax()]
        maxrep = int(rtrace["running"].max()) if rtrace is not None and len(rtrace) else None
        S.append(
            f"\nThroughput peaks at {peak['throughput_rps']:.0f} req/s ({peak['images_per_s']:.0f} images/s) at "
            f"concurrency {int(peak['concurrency'])}. The autoscaler grew the deployment to {maxrep} RUNNING replicas under load "
            "and scaled back down after the downscale delay (Figure 18). Latency at low concurrency is dominated by HTTP, proxy, and "
            "PNG/base64 encoding, not by the roughly 1 ms generator forward pass. At high concurrency, p95/p99 grow because requests "
            "queue while new replicas start; replica start-up costs several seconds (process launch, CUDA context, model load). "
            "Useful production levers: return raw uint8 or binary PNG instead of base64 JSON, raise `max_batch_size`, keep "
            "`min_replicas` >= 2 for availability, and use CPU replicas, since this model is tiny and GPU time is mostly idle.\n")
    S.append(fig("outputs/serve/load_test.png", "Figure 18. Latency, throughput, and replica count during the load test."))

    # ------------------------------------------------------------------ reproducibility
    S.append("## 9. Reproducibility\n")
    S.append(
        "* Every script takes explicit seeds. The Tune search algorithm is seeded. Latents come from a seeded **CPU** "
        "`torch.Generator`, so the same seed gives the same z on any device. Classifier seeds are paired across conditions.\n"
        "* Commands (also run end-to-end in `HW3a_ConditionalGAN.ipynb`):\n"
        "```bash\npython -m pytest -q\npython inspect_data.py\npython train_judge.py\n"
        "python train_gan.py --epochs 60 --out runs/baseline\n"
        "python tune_gan.py --num-samples 2 --max-epochs 2 --smoke\n"
        "python tune_gan.py --num-samples 24 --max-epochs 40 --gpus-per-trial 0.25\n"
        "python evaluate.py --checkpoint artifacts/best_generator.pt --seeds 5\n"
        "python regenerate_grid.py --checkpoint artifacts/best_generator.pt --seed 1234\n```\n"
        f"* Fixed-grid hash check: two independent runs of `regenerate_grid.py` produced "
        f"`{(repro.get('hash_run1') or 'n/a')[:16]}...` and `{(repro.get('hash_run2') or 'n/a')[:16]}...` "
        f"(identical: **{repro.get('identical')}**).\n"
        "* 25 unit tests cover shapes, conditioning, seeding, checkpoint round-trips, metric sanity (copies are flagged, "
        "collapse is detected), API validation, provenance metadata, and batch-invariance.\n")

    # ------------------------------------------------------------------ ethics
    S.append("## 10. Scientific reasoning, ethics, and limitations\n")
    S.append(
        "* **Synthetic is not evidence.** A GAN reproduces the statistics of its training set, including its artefacts. "
        "It cannot produce new clinical knowledge, and a plausible-looking image may show anatomy or pathology that does not "
        "exist (*hallucination*). Synthetic images must never be used as a clinical example, as a test set, or to claim "
        "diagnostic performance.\n"
        "* **Labels are noisy.** Figure 14 shows class-inconsistent samples. A synthetic 'normal' may contain disease-like "
        "patterns. Augmentation with such samples can teach a model wrong decision boundaries.\n"
        "* **Bias is inherited and amplified.** PneumoniaMNIST is derived from one paediatric cohort from a single "
        "centre (Guangzhou; Kermany et al., 2018), at 28x28 resolution, with binary labels that do not separate bacterial from viral pneumonia. "
        "There is no demographic metadata, so subgroup performance cannot be audited. Mode collapse under-samples "
        "rare presentations, which makes the synthetic set *less* diverse than reality.\n"
        "* **Privacy.** Our nearest-neighbour check found no near-copies at the pixel level, but GANs can still leak "
        "membership information. A real deployment would need differential-privacy training or formal membership-inference "
        "audits, and a data-use agreement covering synthetic derivatives.\n"
        "* **Misleading evaluation.** We found that (a) random synthetic images can raise macro-F1 by shifting class priors, "
        "(b) the test split has a different prior from train, and (c) seed variance is comparable to the effects. "
        "Attractive grids or a single run would have supported a false conclusion.\n"
        "* **Misuse.** The API could be used to fabricate 'patient' images for fraud, fake publications, or "
        "insurance claims. Mitigations: an explicit synthetic flag in every response, PNG metadata, and a header; no image upload or "
        "image-to-image editing; a per-request cap; authentication and rate limiting before any shared deployment; logging; "
        "and robust watermarking as future work. Metadata is easy to strip, so it is not a security control.\n"
        "* **Should this connect to clinical workflows?** No. At most it belongs in education (showing what pneumonia-like "
        "opacities look like, with a clear label) and in engineering stress tests (load, format, and pipeline robustness). Any "
        "model trained with synthetic data must be validated on real, external, prospectively collected data.\n\n"
        "See `report/MODEL_CARD.md` for the one-page model card.\n")

    S.append("## References\n")
    S.append(
        "1. Yang et al. *MedMNIST v2*. Scientific Data, 2023. arXiv:2110.14795.\n"
        "2. Radford, Metz, Chintala. *Unsupervised Representation Learning with DCGANs*. arXiv:1511.06434.\n"
        "3. Liaw et al. *Tune: A Research Platform for Distributed Model Selection and Training*. arXiv:1807.05118.\n"
        "4. Goodfellow et al. *Generative Adversarial Nets*. NeurIPS 2014.\n"
        "5. Mirza & Osindero. *Conditional GANs*. arXiv:1411.1784. Miyato & Koyama. *cGANs with Projection Discriminator*. ICLR 2018.\n"
        "6. Miyato et al. *Spectral Normalization for GANs*. ICLR 2018. Li et al. *Hyperband/ASHA*. MLSys 2020.\n"
        "7. Kermany et al. *Identifying Medical Diagnoses ... by Image-Based Deep Learning*. Cell, 2018.\n"
        "8. Ray documentation: Tune, Serve, autoscaling. https://docs.ray.io/en/latest/\n")

    md = "\n\n".join(S)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(md)
    write_html(md, os.path.join(out_dir, "REPORT.html"))
    return md


def write_html(md: str, path: str):
    def embed(m):
        alt, src = m.group(1), m.group(2)
        p = os.path.normpath(os.path.join(os.path.dirname(path), src))
        if os.path.exists(p):
            b64 = base64.b64encode(open(p, "rb").read()).decode()
            return f'<img alt="{alt}" src="data:image/png;base64,{b64}" style="max-width:100%">'
        return m.group(0)

    md2 = re.sub(r"\{width=\"[^\"]*\"\}", "", md)
    try:
        import markdown

        body = markdown.markdown(md2, extensions=["tables", "fenced_code"])
    except ImportError:  # pragma: no cover
        body = "<pre>" + md2 + "</pre>"
    body = re.sub(r'<img alt="([^"]*)" src="([^"]+)" ?/?>', embed, body)
    css = ("body{font-family:Georgia,serif;max-width:900px;margin:2em auto;padding:0 16px;line-height:1.45;color:#222}"
           "table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 6px}"
           "code,pre{background:#f5f5f5;font-size:12px}pre{padding:8px;overflow-x:auto}img{display:block;margin:8px auto}"
           "em{color:#555}@media print{img{max-height:420px}}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'><title>HW3a cGAN report</title>"
                f"<style>{css}</style></head><body>{body}</body></html>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/report")
    a = ap.parse_args()
    build(a.out)
    print("wrote", os.path.join(a.out, "REPORT.md"), "and REPORT.html")
