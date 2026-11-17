"""GAN evaluation beyond 'pretty pictures'.

Signals (all computed on 28x28 round-trip images, see data.py):
* class_consistency_c : fraction of images generated for class c that a frozen judge
                        classifier (trained on real data only) assigns to class c.
* fd_c                : Frechet distance between judge features of real vs fake class c
                        (an FID-style score with a domain-specific feature extractor,
                        because ImageNet Inception features are meaningless for 28x28 CXR).
* div_ratio_c         : mean pairwise judge-feature distance among fakes / among reals of
                        class c. ~1 means real-like diversity; << 1 means mode collapse.
* collapse_penalty    : mean_c max(0, 1 - div_ratio_c).
* memorisation        : nearest-neighbour (pixel RMSE) distance from each fake to the
                        training set, compared with the same distance for *held-out real*
                        validation images. A fake closer to a training image than 95% of
                        real validation images are is flagged. Held-out data already has a
                        5% flag rate by construction, so penalty = excess over 5%.
* val_f1_gain         : macro-F1 on the validation split of a quick classifier trained on
                        real + synthetic-minority minus the same classifier trained on real +
                        the same number of DUPLICATED real minority images (same seeds).
                        The control matters: adding *any* minority-labelled images (even
                        noise) shifts class priors and can raise macro-F1 on an imbalanced
                        split. Comparing against random oversampling isolates the value of
                        the synthetic image *content*.
* quality_score       : val_f1_gain - 0.5 * collapse_penalty - 0.5 * memorization_penalty
"""
from __future__ import annotations

import numpy as np
import torch

from .classifier import compute_metrics, predict_proba, train_classifier
from .data import SplitData, class_counts, from_gan_space, roundtrip
from .models import generate


# ------------------------------------------------------------------ primitive metrics

@torch.no_grad()
def judge_outputs(judge, x01: torch.Tensor, device, batch_size: int = 2048):
    judge.eval()
    feats, probs = [], []
    for i in range(0, len(x01), batch_size):
        f = judge.features(x01[i:i + batch_size].to(device))
        feats.append(f.float().cpu())
        probs.append(torch.softmax(judge.head(f).float(), 1).cpu())
    return torch.cat(feats), torch.cat(probs)


def _sqrtm_psd(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((a + a.T) / 2)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T


def frechet_distance(f1, f2) -> float:
    f1 = np.asarray(f1, dtype=np.float64)
    f2 = np.asarray(f2, dtype=np.float64)
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    r1 = _sqrtm_psd(s1)
    m = r1 @ s2 @ r1
    tr_covmean = np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def mean_pairwise_distance(f: torch.Tensor, max_n: int = 1000, seed: int = 0) -> float:
    f = torch.as_tensor(f, dtype=torch.float32)
    if len(f) > max_n:
        idx = torch.randperm(len(f), generator=torch.Generator().manual_seed(seed))[:max_n]
        f = f[idx]
    n = len(f)
    if n < 2:
        return 0.0
    d = torch.cdist(f, f)
    return float(d.sum() / (n * (n - 1)))


@torch.no_grad()
def nn_distances(query: torch.Tensor, ref: torch.Tensor, chunk: int = 1024):
    """Nearest-neighbour distance (RMSE per pixel) and index from each query row to ref."""
    q = query.flatten(1).float()
    r = ref.flatten(1).float().to(q.device)
    dims = q.shape[1]
    dists, idxs = [], []
    for i in range(0, len(q), chunk):
        d = torch.cdist(q[i:i + chunk], r)
        m, j = d.min(dim=1)
        dists.append(m / dims ** 0.5)
        idxs.append(j)
    return torch.cat(dists).cpu(), torch.cat(idxs).cpu()


def memorization_stats(fake_nn: torch.Tensor, val_nn: torch.Tensor, q: float = 0.05) -> dict:
    thr = float(torch.quantile(val_nn, q))
    rate = float((fake_nn < thr).float().mean())
    return {
        "mem_threshold": thr,
        "mem_rate": rate,
        "memorization_penalty": float(np.clip((rate - q) / (1 - q), 0, 1)),
        "fake_nn_median": float(fake_nn.median()),
        "val_nn_median": float(val_nn.median()),
        "nn_ratio": float(fake_nn.median() / max(1e-8, float(val_nn.median()))),
    }


def quality_score(f1_gain: float, collapse_penalty: float, memorization_penalty: float) -> float:
    return float(f1_gain - 0.5 * collapse_penalty - 0.5 * memorization_penalty)


# ------------------------------------------------------------------------ evaluator

class GANEvaluator:
    """Pre-computes everything that depends only on real data so per-epoch evaluation of
    a generator is cheap enough to run inside every Ray Tune trial."""

    def __init__(self, splits: dict[str, SplitData], judge, device, img_size: int = 64,
                 n_per_class: int = 500, clf_epochs: int = 4, clf_seeds=(0, 1),
                 eval_seed: int = 1234):
        self.device = torch.device(device)
        self.judge = judge.to(self.device).eval()
        self.n = n_per_class
        self.clf_epochs = clf_epochs
        self.clf_seeds = tuple(clf_seeds)
        self.eval_seed = eval_seed
        tr, va = splits["train"], splits["val"]
        self.x_tr, self.y_tr = roundtrip(tr.x28, img_size), tr.y.clone()
        self.x_va, self.y_va = roundtrip(va.x28, img_size), va.y.clone()

        feats, _ = judge_outputs(self.judge, self.x_tr, self.device)
        self.real_feats = {c: feats[self.y_tr == c] for c in (0, 1)}
        self.real_div = {c: mean_pairwise_distance(self.real_feats[c], max_n=n_per_class) for c in (0, 1)}
        self.real_pix_div = {c: mean_pairwise_distance(self.x_tr[self.y_tr == c].flatten(1), max_n=n_per_class)
                             for c in (0, 1)}
        self.train_flat = self.x_tr.flatten(1).to(self.device)
        self.val_nn, _ = nn_distances(self.x_va.to(self.device), self.train_flat)

        counts = class_counts(self.y_tr)
        self.minority = int(np.argmin(counts))
        self.n_syn = int(max(counts) - min(counts))
        self._base_f1 = None

    # -------------------------------------------------------------------------------
    def _quick_f1(self, x_extra=None, y_extra=None, seed: int = 0) -> float:
        x, y = self.x_tr, self.y_tr
        if x_extra is not None:
            x, y = torch.cat([x, x_extra]), torch.cat([y, y_extra])
        model, _ = train_classifier(x, y, self.device, epochs=self.clf_epochs, seed=seed, batch_size=256)
        return compute_metrics(self.y_va.numpy(), predict_proba(model, self.x_va, self.device))["f1_macro"]

    @property
    def base_f1(self) -> float:
        if self._base_f1 is None:
            with torch.random.fork_rng(devices=self._rng_devices()):
                g = torch.Generator().manual_seed(self.eval_seed)
                mi = torch.nonzero(self.y_tr == self.minority).squeeze(1)
                dup = mi[torch.randint(len(mi), (self.n_syn,), generator=g)]
                self._base_f1 = float(np.mean([self._quick_f1(self.x_tr[dup], self.y_tr[dup], seed=s)
                                               for s in self.clf_seeds]))
        return self._base_f1

    def _rng_devices(self):
        if self.device.type == "cuda":
            return [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        return []

    @torch.no_grad()
    def generate28(self, G, labels, seed: int) -> torch.Tensor:
        return from_gan_space(generate(G, labels, seed, self.device))

    def evaluate(self, G, downstream: bool = True) -> dict:
        was_training = G.training
        G.eval()
        try:
            with torch.random.fork_rng(devices=self._rng_devices()):
                return self._evaluate(G, downstream)
        finally:
            G.train(was_training)

    def _evaluate(self, G, downstream: bool) -> dict:
        labels = torch.tensor([0] * self.n + [1] * self.n)
        fake = self.generate28(G, labels, self.eval_seed)
        out: dict[str, float] = {}
        if not torch.isfinite(fake).all():
            return {"quality_score": -1.0, "nonfinite": 1.0}
        feats, probs = judge_outputs(self.judge, fake, self.device)
        collapse = []
        for c in (0, 1):
            m = labels == c
            out[f"class_consistency_{c}"] = float((probs[m].argmax(1) == c).float().mean())
            ref = self.real_feats[c]
            if len(ref) > 2000:
                ref = ref[torch.randperm(len(ref), generator=torch.Generator().manual_seed(0))[:2000]]
            out[f"fd_{c}"] = frechet_distance(feats[m].numpy(), ref.numpy())
            out[f"div_ratio_{c}"] = mean_pairwise_distance(feats[m]) / max(1e-8, self.real_div[c])
            out[f"pix_div_ratio_{c}"] = mean_pairwise_distance(fake[m].flatten(1)) / max(1e-8, self.real_pix_div[c])
            collapse.append(max(0.0, 1.0 - out[f"div_ratio_{c}"]))
        out["class_consistency"] = (out["class_consistency_0"] + out["class_consistency_1"]) / 2
        out["fd_mean"] = (out["fd_0"] + out["fd_1"]) / 2
        out["collapse_penalty"] = float(np.mean(collapse))
        fake_nn, _ = nn_distances(fake.to(self.device), self.train_flat)
        out.update(memorization_stats(fake_nn, self.val_nn))

        if downstream:
            syn = self.generate28(G, [self.minority] * self.n_syn, self.eval_seed + 1)
            ys = torch.full((self.n_syn,), self.minority, dtype=torch.long)
            f1s = [self._quick_f1(syn, ys, seed=s) for s in self.clf_seeds]
            out["val_f1_aug"] = float(np.mean(f1s))
            out["val_f1_base"] = self.base_f1
            out["val_f1_gain"] = out["val_f1_aug"] - out["val_f1_base"]
        else:
            out["val_f1_gain"] = 0.0
        out["quality_score"] = quality_score(out["val_f1_gain"], out["collapse_penalty"],
                                             out["memorization_penalty"])
        return out
