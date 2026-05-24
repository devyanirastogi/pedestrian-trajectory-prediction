"""
cvae.py — Conditional VAE decoder, the ablation against our diffusion decoder.

Reference: the MID paper, Table 3 (group 4): "diffusion decoder vs CVAE decoder,
*same encoder*." So we keep the Trajectron++ encoder (which produces the latent
f) and swap our DiffusionTraj for a CVAE that decodes f → future trajectory.

Architecture:

    Future encoder  : LSTM over the ground-truth future trajectory y₀ (used at
                      training time only to compute the posterior).
        y (B, T, 2)  → h_y (B, h_future)

    Prior p(z|f)    : MLP from the encoder latent.
        f (B, F)     → μ_p (B, z_dim), log σ_p (B, z_dim)

    Posterior q(z|f, h_y) : MLP on [f, h_y].
        (f, h_y)     → μ_q (B, z_dim), log σ_q (B, z_dim)

    Decoder p(y|z, f) : MLP that maps [z, f] to a flat future trajectory,
                        reshaped to (B, T, 2).

Loss (per-batch):

    ELBO  = MSE(ŷ, y) + β · KL[ q(z|f, y) ‖ p(z|f) ]

We use the *learned* conditional prior p(z|f) rather than N(0, I), matching
MGCVAE's structure. β is the KL weight; β=1 is the proper ELBO, smaller
values trade calibration for reconstruction quality.

Inference (Best-of-K):
    Sample K times: z_k ~ p(z|f), then ŷ_k = decoder(z_k, f).
    The K sampled futures plug straight into the existing eval pipeline.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# Sub-networks
# ─────────────────────────────────────────────────────────────────────
class FutureEncoder(nn.Module):
    """LSTM that summarizes the ground-truth future trajectory.

    Used only during training (to compute the posterior). At inference we
    don't see the future, so we sample from p(z|f) instead.
    """

    def __init__(self, point_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(point_dim, hidden_dim, batch_first=True)
        self.out_dim = hidden_dim

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, T, 2) → take the final hidden state
        _, (h_n, _) = self.lstm(y)        # h_n: (1, B, hidden)
        return h_n.squeeze(0)              # (B, hidden)


class GaussianHead(nn.Module):
    """Maps a feature vector to a diagonal-Gaussian (μ, log σ)."""

    def __init__(self, in_dim: int, hidden_dim: int, z_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        # Clamp log σ to avoid numerical blow-ups (KL with σ→0 or σ→∞)
        return self.mu(h), self.log_sigma(h).clamp(-6.0, 2.0)


class Decoder(nn.Module):
    """MLP decoder: [z, f] → flat future trajectory, reshape to (B, T, point_dim)."""

    def __init__(
        self,
        z_dim: int,
        context_dim: int,
        hidden_dim: int = 256,
        future_steps: int = 12,
        point_dim: int = 2,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.point_dim = point_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, future_steps * point_dim),
        )

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # z: (B, z_dim) or (S, B, z_dim) — we broadcast context the same way
        if z.dim() == 3:
            S, B, _ = z.shape
            ctx = context.unsqueeze(0).expand(S, B, -1)
            inp = torch.cat([z, ctx], dim=-1)               # (S, B, z+f)
            out = self.net(inp)                              # (S, B, T*2)
            return out.view(S, B, self.future_steps, self.point_dim)
        # 2D path: single sample per batch element
        inp = torch.cat([z, context], dim=-1)                # (B, z+f)
        out = self.net(inp)                                  # (B, T*2)
        return out.view(-1, self.future_steps, self.point_dim)


# ─────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────
def kl_diag_gaussian(
    mu_q: torch.Tensor, log_sigma_q: torch.Tensor,
    mu_p: torch.Tensor, log_sigma_p: torch.Tensor,
) -> torch.Tensor:
    """
    KL[ N(μ_q, σ_q²) ‖ N(μ_p, σ_p²) ] for diagonal-covariance Gaussians.
    Returns a (B,) tensor — sum over z dimensions, mean taken by caller.
    """
    sigma_q_sq = (2 * log_sigma_q).exp()
    sigma_p_sq = (2 * log_sigma_p).exp()
    return 0.5 * (
        (log_sigma_p - log_sigma_q) * 2
        + (sigma_q_sq + (mu_q - mu_p).pow(2)) / sigma_p_sq
        - 1
    ).sum(dim=-1)


def reparameterize(mu: torch.Tensor, log_sigma: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
    """
    Diagonal-Gaussian reparameterization trick.

    If n_samples=1 returns (B, z_dim); otherwise (n_samples, B, z_dim) for
    drawing multiple samples efficiently (used at inference for Best-of-K).
    """
    sigma = log_sigma.exp()
    if n_samples == 1:
        eps = torch.randn_like(mu)
        return mu + sigma * eps
    eps = torch.randn(n_samples, *mu.shape, device=mu.device, dtype=mu.dtype)
    return mu.unsqueeze(0) + sigma.unsqueeze(0) * eps


# ─────────────────────────────────────────────────────────────────────
# CVAE wrapper (mirrors DiffusionTraj's get_loss / sample interface)
# ─────────────────────────────────────────────────────────────────────
class CVAETraj(nn.Module):
    """
    Conditional VAE that takes the encoder latent f as context and outputs a
    future trajectory. Drop-in API replacement for DiffusionTraj so the rest
    of the pipeline (AutoEncoder, training loop, eval) doesn't change shape.

    Shapes:
        context (f): (B, context_dim)
        y_0:         (B, T, point_dim) — typically (B, 12, 2) of velocity

    Loss interface:
        get_loss(y_0, context) → scalar (recon MSE + β·KL)

    Sampling interface:
        sample(num_points, context, sample=K, bestof=True, ...) → (K, B, T, 2)
    """

    def __init__(
        self,
        context_dim: int = 256,
        z_dim: int = 32,
        future_hidden: int = 128,
        prior_hidden: int = 64,
        posterior_hidden: int = 64,
        decoder_hidden: int = 256,
        future_steps: int = 12,
        point_dim: int = 2,
        kl_weight: float = 1.0,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.future_steps = future_steps
        self.kl_weight = kl_weight

        self.future_encoder = FutureEncoder(point_dim=point_dim, hidden_dim=future_hidden)
        self.prior = GaussianHead(in_dim=context_dim, hidden_dim=prior_hidden, z_dim=z_dim)
        self.posterior = GaussianHead(
            in_dim=context_dim + future_hidden,
            hidden_dim=posterior_hidden,
            z_dim=z_dim,
        )
        self.decoder = Decoder(
            z_dim=z_dim,
            context_dim=context_dim,
            hidden_dim=decoder_hidden,
            future_steps=future_steps,
            point_dim=point_dim,
        )

    def get_loss(self, x_0: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        ELBO = MSE(ŷ, y) + β · KL[ q(z|f, y) ‖ p(z|f) ].

        x_0:     ground-truth future, shape (B, T, point_dim). Named x_0 for
                 symmetry with DiffusionTraj.get_loss — semantically it's y₀.
        context: encoder latent f, shape (B, context_dim).
        """
        # Posterior: condition on (f, encoded future)
        h_y = self.future_encoder(x_0)                       # (B, future_hidden)
        mu_q, log_sigma_q = self.posterior(torch.cat([context, h_y], dim=-1))

        # Prior: condition on f only
        mu_p, log_sigma_p = self.prior(context)

        # Reparam sample
        z = reparameterize(mu_q, log_sigma_q, n_samples=1)   # (B, z_dim)

        # Decode and compute reconstruction loss
        y_hat = self.decoder(z, context)                     # (B, T, point_dim)
        recon = F.mse_loss(y_hat, x_0, reduction="mean")

        # KL term — mean over the batch
        kl = kl_diag_gaussian(mu_q, log_sigma_q, mu_p, log_sigma_p).mean()

        return recon + self.kl_weight * kl

    @torch.no_grad()
    def sample(
        self,
        num_points: int,
        context: torch.Tensor,
        sample: int = 20,
        bestof: bool = True,
        point_dim: int = 2,
        flexibility: float = 0.0,
        ret_traj: bool = False,
        sampling: str = "ddpm",
        step: int = 1,
    ) -> torch.Tensor:
        """
        Draw `sample` futures per batch element by sampling z ~ p(z|f) and
        decoding. Signature matches DiffusionTraj.sample so the AutoEncoder's
        generate() and the eval pipeline work unchanged.

        The diffusion-specific kwargs (flexibility, sampling, step, ret_traj)
        are accepted-and-ignored so this is a true drop-in.
        """
        del flexibility, ret_traj, sampling, step  # diffusion-only knobs
        del point_dim  # we always use self.point_dim

        # Override num_points if caller passed something else; CVAE's decoder
        # has a fixed output length.
        if num_points != self.future_steps:
            raise ValueError(
                f"CVAETraj was built for future_steps={self.future_steps}, "
                f"got num_points={num_points}"
            )

        mu_p, log_sigma_p = self.prior(context)              # (B, z_dim) each
        if bestof:
            z = reparameterize(mu_p, log_sigma_p, n_samples=sample)  # (S, B, z_dim)
        else:
            # Deterministic mean of the prior, replicated across S.
            z = mu_p.unsqueeze(0).expand(sample, -1, -1)

        y_hat = self.decoder(z, context)                     # (S, B, T, 2)
        return y_hat
