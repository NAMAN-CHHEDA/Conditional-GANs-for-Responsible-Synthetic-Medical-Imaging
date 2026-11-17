import torch

from gan_lab.models import Discriminator, Generator, SmallCNN, build_gan, generate, sample_latent
from gan_lab.trainer import DEFAULT_CONFIG, GANTrainer


def test_generator_shapes_and_range():
    for size in (32, 64):
        G = Generator(latent_dim=16, ngf=8, img_size=size).eval()
        x = G(torch.randn(3, 16), torch.tensor([0, 1, 0]))
        assert x.shape == (3, 1, size, size)
        assert x.min() >= -1 and x.max() <= 1


def test_discriminator_projection_and_sn():
    for sn in (False, True):
        D = Discriminator(ndf=8, img_size=64, spectral_norm=sn)
        out = D(torch.randn(4, 1, 64, 64), torch.tensor([0, 1, 1, 0]))
        assert out.shape == (4,)
        assert torch.isfinite(out).all()


def test_label_conditions_output():
    G = Generator(latent_dim=16, ngf=8).eval()
    z = torch.randn(2, 16)
    a = G(z, torch.tensor([0, 0]))
    b = G(z, torch.tensor([1, 1]))
    assert not torch.allclose(a, b), "changing the label must change the image"


def test_seeded_generation_is_reproducible():
    G = Generator(latent_dim=16, ngf=8).eval()
    assert torch.equal(sample_latent(5, 16, 3), sample_latent(5, 16, 3))
    assert torch.equal(generate(G, [0, 1, 1], seed=11), generate(G, [0, 1, 1], seed=11))
    assert not torch.equal(generate(G, [0, 1, 1], seed=11), generate(G, [0, 1, 1], seed=12))


def test_eval_mode_output_independent_of_batch():
    """Serve batching relies on this: in eval mode a sample does not depend on its batch-mates."""
    G = Generator(latent_dim=16, ngf=8).eval()
    z = torch.randn(4, 16)
    y = torch.tensor([0, 1, 0, 1])
    with torch.no_grad():
        full = G(z, y)
        single = G(z[:1], y[:1])
    assert torch.allclose(full[:1], single, atol=1e-5)


def test_train_epoch_and_checkpoint_roundtrip():
    cfg = {**DEFAULT_CONFIG, "latent_dim": 16, "ngf": 8, "ndf": 8, "batch_size": 8}
    tr = GANTrainer(cfg, "cpu")
    x = torch.rand(32, 1, 64, 64) * 2 - 1
    y = torch.randint(0, 2, (32,))
    stats = tr.train_epoch(x, y)
    for k in ("d_loss", "g_loss", "d_acc"):
        assert torch.isfinite(torch.tensor(stats[k]))
    state = tr.state_dict()
    tr2 = GANTrainer(cfg, "cpu")
    tr2.load_state_dict(state)
    tr.G.eval()
    tr2.G.eval()
    assert torch.equal(generate(tr.G, [0, 1], 5), generate(tr2.G, [0, 1], 5))


def test_small_cnn():
    m = SmallCNN()
    assert m(torch.rand(5, 1, 28, 28)).shape == (5, 2)
    assert m.features(torch.rand(5, 1, 28, 28)).shape == (5, 128)


def test_build_gan_from_config():
    G, D = build_gan({**DEFAULT_CONFIG, "ngf": 8, "ndf": 8})
    assert G.img_size == 64
