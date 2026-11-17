"""Adversarial training loop for the conditional DCGAN (shared by the baseline script and
the Ray Tune trainable)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .models import build_gan, count_params

DEFAULT_CONFIG = {
    "img_size": 64,
    "latent_dim": 64,
    "embed_dim": 32,
    "ngf": 64,
    "ndf": 64,
    "lr_g": 2e-4,
    "lr_d": 2e-4,
    "beta1": 0.5,
    "beta2": 0.999,
    "batch_size": 64,
    "label_smoothing": 0.1,   # one-sided: real target = 1 - label_smoothing
    "d_steps": 1,             # discriminator updates per generator update
    "spectral_norm": False,
    "seed": 0,
}


class GANTrainer:
    def __init__(self, cfg: dict, device):
        self.cfg = {**DEFAULT_CONFIG, **cfg}
        self.device = torch.device(device)
        self.G, self.D = build_gan(self.cfg)
        self.G.to(self.device)
        self.D.to(self.device)
        betas = (self.cfg["beta1"], self.cfg["beta2"])
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=self.cfg["lr_g"], betas=betas)
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=self.cfg["lr_d"], betas=betas)
        self.bce = nn.BCEWithLogitsLoss()
        self.epoch = 0
        self.step = 0

    def param_counts(self) -> dict:
        return {"G": count_params(self.G), "D": count_params(self.D)}

    def train_epoch(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """One pass over (x, y) which must already be in GAN space and on self.device."""
        cfg, G, D = self.cfg, self.G, self.D
        G.train()
        D.train()
        bs, n = int(cfg["batch_size"]), len(y)
        real_t = 1.0 - float(cfg["label_smoothing"])
        perm = torch.randperm(n, device=x.device)
        sums = {"d_loss": 0.0, "g_loss": 0.0, "d_acc": 0.0, "d_acc_real": 0.0, "d_acc_fake": 0.0,
                "d_real": 0.0, "d_fake": 0.0, "g_grad_norm": 0.0}
        n_batches = 0
        for i in range(0, n - bs + 1, bs):  # drop_last keeps BN statistics well-defined
            idx = perm[i:i + bs]
            real, lab = x[idx], y[idx]
            # ---- discriminator: maximise log D(x|y) + log(1 - D(G(z|y)|y))
            for _ in range(int(cfg["d_steps"])):
                z = torch.randn(bs, G.latent_dim, device=x.device)
                fake = G(z, lab)
                real_logits = D(real, lab)
                fake_logits = D(fake.detach(), lab)  # detach: no generator gradients here
                d_loss = (self.bce(real_logits, torch.full_like(real_logits, real_t))
                          + self.bce(fake_logits, torch.zeros_like(fake_logits)))
                self.opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                self.opt_d.step()
            # ---- generator: non-saturating loss, maximise log D(G(z|y)|y)
            z = torch.randn(bs, G.latent_dim, device=x.device)
            fake = G(z, lab)
            g_logits = D(fake, lab)
            g_loss = self.bce(g_logits, torch.ones_like(g_logits))
            self.opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            gn = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in G.parameters() if p.grad is not None))
            self.opt_g.step()

            with torch.no_grad():
                acc_r = (real_logits > 0).float().mean().item()
                acc_f = (fake_logits < 0).float().mean().item()
                sums["d_loss"] += d_loss.item()
                sums["g_loss"] += g_loss.item()
                sums["d_acc_real"] += acc_r
                sums["d_acc_fake"] += acc_f
                sums["d_acc"] += (acc_r + acc_f) / 2
                sums["d_real"] += torch.sigmoid(real_logits).mean().item()
                sums["d_fake"] += torch.sigmoid(fake_logits).mean().item()
                sums["g_grad_norm"] += gn.item()
            n_batches += 1
            self.step += 1
        self.epoch += 1
        return {k: v / max(1, n_batches) for k, v in sums.items()}

    # ----------------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {"config": self.cfg, "G": self.G.state_dict(), "D": self.D.state_dict(),
                "opt_g": self.opt_g.state_dict(), "opt_d": self.opt_d.state_dict(),
                "epoch": self.epoch, "step": self.step}

    def load_state_dict(self, s: dict) -> None:
        self.G.load_state_dict(s["G"])
        self.D.load_state_dict(s["D"])
        self.opt_g.load_state_dict(s["opt_g"])
        self.opt_d.load_state_dict(s["opt_d"])
        self.epoch, self.step = s.get("epoch", 0), s.get("step", 0)
