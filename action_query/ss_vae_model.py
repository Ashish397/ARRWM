"""Semi-supervised beta-VAE for disentangled motion encoding.

Extracted from action_query/disentangled_vae_clustering.ipynb so that
utils/test_zarr_chunks.py (and other scripts) can import it standalone.
"""

from pathlib import Path

import torch
import torch.nn as nn


class FullyConvEncoder(nn.Module):
    def __init__(self, in_ch: int = 2, latent_ch: int = 16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16,   32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32,   32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.conv_mu     = nn.Conv2d(32, latent_ch, 3, stride=1, padding=0)
        self.conv_logvar = nn.Conv2d(32, latent_ch, 3, stride=1, padding=0)

    def forward(self, x):
        h = self.backbone(x)
        return self.conv_mu(h), self.conv_logvar(h)


class SplitDecoder(nn.Module):
    def __init__(self, n_sup=4, n_free=12, out_ch=2):
        super().__init__()
        self.n_sup = n_sup

        def _head(in_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, 16, 3, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
            )

        if n_sup > 0:
            self.sup_head = _head(n_sup)
        self.free_head = _head(n_free)
        merge_ch = (8 if n_sup > 0 else 0) + 8
        self.out = nn.Conv2d(merge_ch, out_ch, 3, stride=1, padding=1)

    def forward(self, z, n_sup=None):
        n_sup = n_sup if n_sup is not None else self.n_sup
        z_free = z[:, n_sup:]
        if n_sup > 0:
            z_sup = z[:, :n_sup]
            return self.out(torch.cat([self.sup_head(z_sup), self.free_head(z_free)], dim=1))
        return self.out(self.free_head(z_free))


class SemiSupervisedBetaVAE(nn.Module):
    def __init__(self, in_ch: int = 2, latent_ch: int = 16,
                 n_sup: int = 5, n_free: int = 11):
        super().__init__()
        self.encoder = FullyConvEncoder(in_ch, latent_ch)
        self.decoder = SplitDecoder(n_sup=n_sup, n_free=n_free, out_ch=in_ch)
        self.latent_ch = latent_ch
        self.n_sup = n_sup

    def reparameterise(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x, sample: bool = True):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar) if sample else mu
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z


def load_ss_vae(checkpoint_path: Path | str, device: str = "cpu"):
    """Load a SemiSupervisedBetaVAE from a checkpoint saved by the notebook.

    Returns (model, scale) where scale is the normalisation factor used during
    training (divide raw motion dx/dy by this before feeding to the encoder).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hp = ckpt["hparams"]
    model = SemiSupervisedBetaVAE(
        in_ch=2,
        latent_ch=hp["LATENT_CH"],
        n_sup=hp["N_SUPERVISED"],
        n_free=hp["N_FREE"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scale = float(ckpt["scale"])
    return model, scale
