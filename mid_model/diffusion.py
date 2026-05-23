"""
diffusion.py — DDPM forward/reverse process + Transformer noise predictor.

Clean reimplementation of MID's diffusion stack (only the TransformerConcatLinear
path that the paper actually uses). Fixes a couple of warts from the original:

    1. No hardcoded `.cuda()` calls — uses the device of the input tensors so
       this works on CPU, MPS (Apple Silicon), and CUDA.
    2. Tensor ops use `torch.gather` / advanced indexing in a device-agnostic
       way; no implicit host transfers.

Three classes:

    VarianceSchedule
        Holds the β / α / ᾱ buffers for the K-step diffusion process.
        Linear schedule from β₁=1e-4 to β_K=5e-2 (K=100 in the paper).

    TransformerConcatLinear
        ε-prediction network. Input: noised trajectory y_k, noise level β_k,
        encoder context f. Output: predicted noise ε̂. Architecture is
        unchanged from MID — the only model class we reimplement here.

    DiffusionTraj
        Wraps a noise predictor + a schedule. .get_loss(y0, context) implements
        the standard DDPM training objective; .sample(...) runs the reverse
        chain (DDPM or DDIM) to generate trajectory samples from noise.

Notation matches the DDPM paper (Ho et al. 2020):
    α_k    = 1 - β_k
    ᾱ_k    = ∏_{i=1..k} α_i      ("alpha_bar")
    σ_k    = √β_k                (DDPM σ, the "flexible" choice)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .common import ConcatSquashLinear, PositionalEncoding


# ─────────────────────────────────────────────────────────────────────
# Variance schedule
# ─────────────────────────────────────────────────────────────────────
class VarianceSchedule(nn.Module):
    """
    Precomputes the β / α / ᾱ / σ buffers for a K-step DDPM process.

    Convention: indices run 1..K (not 0..K-1). Index 0 is padded so that
    `betas[0]=0`, `alphas[0]=1`, `alpha_bars[0]=1`. This makes the reverse
    step's final iteration (k=1 → k=0) clean: at k=0 we're at "clean data".
    """

    def __init__(
        self,
        num_steps: int = 100,
        mode: str = "linear",
        beta_1: float = 1e-4,
        beta_T: float = 5e-2,
        cosine_s: float = 8e-3,
    ):
        super().__init__()
        assert mode in ("linear", "cosine")
        self.num_steps = num_steps
        self.mode = mode

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        else:  # cosine schedule (Nichol & Dhariwal 2021)
            timesteps = (torch.arange(num_steps + 1) / num_steps + cosine_s)
            alphas = timesteps / (1 + cosine_s) * np.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        # Pad index 0 so we can index by t = 1..K directly.
        betas = torch.cat([torch.zeros(1), betas], dim=0)  # (K+1,)

        alphas = 1 - betas
        # Cumulative product of alphas, computed in log space for stability.
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        # Two choices of reverse-process σ (Ho et al. 2020, eq. 7).
        # "flex"  : σ_k = √β_k                          — higher variance
        # "inflex": σ_k = √((1-ᾱ_{k-1})/(1-ᾱ_k) · β_k)  — lower variance
        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size: int):
        """Sample one k ~ Uniform{1, ..., K} per element in the batch."""
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility: float = 0.0):
        """Interpolate between the two σ choices: 0 = inflex (low var), 1 = flex (high var)."""
        assert 0.0 <= flexibility <= 1.0
        return self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)


# ─────────────────────────────────────────────────────────────────────
# Noise predictor (the only architecture we use from the paper)
# ─────────────────────────────────────────────────────────────────────
class TransformerConcatLinear(nn.Module):
    """
    ε-prediction network. The paper's chosen decoder architecture.

    Input:
        x:       noised trajectory y_k, shape (B, T, 2)   where T=12
        beta:    diffusion noise level β_k, shape (B,)
        context: encoder latent f, shape (B, F)           where F=256

    The context fed into every ConcatSquashLinear is a concatenation:
        ctx = [β, sin(β), cos(β), f]   shape (B, 1, F+3)
    This gives the layers a richer time embedding than raw β alone — the
    sin/cos pair makes nearby noise levels distinguishable across orders of
    magnitude (β ranges from 1e-4 to 5e-2).

    Pipeline:
        (B, T, 2) → ConcatSquashLinear → (B, T, 2F)
                  → +positional encoding
                  → 3-layer TransformerEncoder
                  → ConcatSquashLinear → (B, T, F)
                  → ConcatSquashLinear → (B, T, F/2)
                  → ConcatSquashLinear → (B, T, 2)         ← predicted ε

    Note: the Transformer here is plain self-attention over the 12 timesteps.
    No causal mask — at noise level k we're allowed to look at the whole
    trajectory bidirectionally, which is correct for non-autoregressive
    denoising.
    """

    def __init__(
        self,
        point_dim: int = 2,
        context_dim: int = 256,
        tf_layer: int = 3,
        residual: bool = False,
    ):
        super().__init__()
        self.residual = residual
        ctx_in = context_dim + 3  # +3 for [β, sin(β), cos(β)]

        d_model = 2 * context_dim  # 512 for the paper's encoder_dim=256

        self.concat1 = ConcatSquashLinear(point_dim, d_model, ctx_in)
        self.pos_emb = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=24)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * context_dim,  # 1024 in the paper
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=tf_layer)

        # Downsample: 2F → F → F/2 → 2
        self.concat3 = ConcatSquashLinear(d_model, context_dim, ctx_in)
        self.concat4 = ConcatSquashLinear(context_dim, context_dim // 2, ctx_in)
        self.linear  = ConcatSquashLinear(context_dim // 2, point_dim, ctx_in)

    def forward(self, x: torch.Tensor, beta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x:       (B, T, 2)
        # beta:    (B,)    — one noise level per batch element
        # context: (B, F)
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)              # (B, 1, 1)
        context = context.view(batch_size, 1, -1)      # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)                         # (B, 1, F+3)

        x = self.concat1(ctx_emb, x)                    # (B, T, 2F)

        # TransformerEncoder expects (T, B, d_model). Permute, encode, permute back.
        x = x.permute(1, 0, 2)                          # (T, B, 2F)
        x = self.pos_emb(x)
        x = self.transformer_encoder(x)                 # (T, B, 2F)
        x = x.permute(1, 0, 2)                          # (B, T, 2F)

        x = self.concat3(ctx_emb, x)                    # (B, T, F)
        x = self.concat4(ctx_emb, x)                    # (B, T, F/2)
        return self.linear(ctx_emb, x)                  # (B, T, 2)


# ─────────────────────────────────────────────────────────────────────
# Diffusion process wrapper (forward noising + reverse sampling)
# ─────────────────────────────────────────────────────────────────────
class DiffusionTraj(nn.Module):
    """
    Wires a noise predictor `net` together with a `VarianceSchedule`.

    Training:
        loss = E_{y₀, k, ε} || ε - ε̂(√ᾱ_k y₀ + √(1-ᾱ_k) ε, β_k, f) ||²
        That's the standard DDPM ε-prediction objective.

    Sampling:
        y_K ~ N(0, I), then iteratively denoise via either DDPM or DDIM updates.
    """

    def __init__(self, net: nn.Module, var_sched: VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(
        self,
        x_0: torch.Tensor,
        context: torch.Tensor,
        t: Optional[list] = None,
    ) -> torch.Tensor:
        # x_0:     clean trajectory y₀, shape (B, T, point_dim)
        # context: encoder latent f, shape (B, F)
        batch_size, _, point_dim = x_0.size()
        device = x_0.device

        if t is None:
            t = self.var_sched.uniform_sample_t(batch_size)

        # Pull the schedule values for this batch's noise levels.
        # The buffers live on the schedule module's device — they get moved
        # along with the model via .to(device).
        alpha_bar = self.var_sched.alpha_bars[t].to(device)  # (B,)
        beta = self.var_sched.betas[t].to(device)            # (B,)

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)        # (B, 1, 1)  scales y₀
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)    # (B, 1, 1)  scales noise

        e_rand = torch.randn_like(x_0)                   # ε ~ N(0, I)

        # Forward noising in closed form: y_k = √ᾱ_k · y₀ + √(1-ᾱ_k) · ε
        y_k = c0 * x_0 + c1 * e_rand
        e_theta = self.net(y_k, beta=beta, context=context)

        return F.mse_loss(e_theta.reshape(-1, point_dim), e_rand.reshape(-1, point_dim), reduction="mean")

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
        Generate `sample` trajectories for each item in the batch.

        Args:
            num_points:  T (number of future timesteps to predict, e.g. 12)
            context:     (B, F)  encoder latents
            sample:      number of independent samples to draw per batch element
                         (Best-of-20 → sample=20)
            bestof:      if True start from y_K ~ N(0, I); if False from zeros
            sampling:    "ddpm" or "ddim"
            step:        stride for the reverse chain. step=1 = full K steps
                         (paper default). step=K reduces to a single step.

        Returns:
            tensor of shape (sample, B, T, point_dim) if ret_traj=False,
            otherwise a list of {timestep: tensor} dicts.
        """
        device = context.device
        K = self.var_sched.num_steps
        traj_list = []

        for _ in range(sample):
            batch_size = context.size(0)
            if bestof:
                x_T = torch.randn(batch_size, num_points, point_dim, device=device)
            else:
                x_T = torch.zeros(batch_size, num_points, point_dim, device=device)

            traj = {K: x_T}

            for t in range(K, 0, -step):
                # z ~ N(0, I) for stochastic step; z=0 at the very last step
                # to produce a deterministic clean output.
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                alpha_bar_next = self.var_sched.alpha_bars[max(t - step, 0)]
                sigma = self.var_sched.get_sigmas(t, flexibility)

                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                x_t = traj[t]
                beta = self.var_sched.betas[[t] * batch_size].to(device)
                e_theta = self.net(x_t, beta=beta, context=context)

                if sampling == "ddpm":
                    # Standard DDPM update (Ho et al. 2020, eq. 11)
                    x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                elif sampling == "ddim":
                    # Deterministic DDIM update (Song et al. 2020)
                    x0_pred = (x_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x_next = alpha_bar_next.sqrt() * x0_pred + (1 - alpha_bar_next).sqrt() * e_theta
                else:
                    raise ValueError(f"unknown sampling mode: {sampling}")

                traj[max(t - step, 0)] = x_next.detach()
                traj[t] = traj[t].cpu()  # free GPU memory as we go
                if not ret_traj:
                    del traj[t]

            traj_list.append(traj if ret_traj else traj[0])

        if ret_traj:
            return traj_list
        return torch.stack(traj_list)  # (sample, B, T, point_dim)
