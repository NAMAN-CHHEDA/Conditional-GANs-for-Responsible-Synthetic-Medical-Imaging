import numpy as np
import torch

from gan_lab.classifier import compute_metrics, random_augment
from gan_lab.data import from_gan_space, roundtrip, to_gan_space
from gan_lab.metrics import (frechet_distance, mean_pairwise_distance, memorization_stats, nn_distances,
                             quality_score)


def _data(n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 1, 28, 28, generator=g)


def test_memorization_detects_copies():
    train, val = _data(400, 0), _data(100, 1)
    val_nn, _ = nn_distances(val, train)
    copies = train[:200] + 0.001 * torch.randn(200, 1, 28, 28)
    fake_nn, idx = nn_distances(copies, train)
    assert torch.equal(idx, torch.arange(200))
    assert memorization_stats(fake_nn, val_nn)["memorization_penalty"] > 0.9
    fresh_nn, _ = nn_distances(_data(200, 2), train)
    assert memorization_stats(fresh_nn, val_nn)["memorization_penalty"] < 0.1


def test_collapse_proxy():
    f = torch.randn(200, 16)
    collapsed = f[:1].repeat(200, 1)
    assert mean_pairwise_distance(collapsed) < 1e-2
    assert mean_pairwise_distance(f) > 1.0


def test_frechet_distance():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2000, 8))
    b = rng.normal(size=(2000, 8))
    assert frechet_distance(a, b) < 0.1
    assert frechet_distance(a, b + 3) > 8 * 9 * 0.9  # mean shift of 3 in 8 dims ~ 72


def test_quality_score_formula():
    assert quality_score(0.02, 0.1, 0.2) == 0.02 - 0.05 - 0.1


def test_space_conversions():
    x = _data(4)
    g = to_gan_space(x, 64)
    assert g.shape[-1] == 64 and g.min() >= -1 and g.max() <= 1
    back = from_gan_space(g, 28)
    assert back.shape == x.shape
    smooth = torch.linspace(0, 1, 28).view(1, 1, 1, 28).expand(2, 1, 28, 28).contiguous()
    assert (roundtrip(smooth, 64) - smooth).abs().mean() < 0.02  # near-lossless on smooth images


def test_compute_metrics_and_augment():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.6, 0.7, 0.9]))
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert abs(m["balanced_accuracy"] - 0.75) < 1e-9
    xa = random_augment(_data(8))
    assert xa.shape == (8, 1, 28, 28) and xa.min() >= 0 and xa.max() <= 1
