"""
common.py — building blocks shared by the diffusion noise-prediction network.

This file is a clean reimplementation of the two layers from MID/models/common.py
that we actually need for TransformerConcatLinear:

    - PositionalEncoding  : standard sinusoidal positional embeddings
    - ConcatSquashLinear  : context-conditioned linear layer (FiLM-like gating)

Everything else in MID's common.py (reparameterize_gaussian, gaussian_entropy,
truncated_normal_, ConcatTransformerLinear, lr_func helpers) is dead code for
the TransformerConcatLinear path and we omit it.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from "Attention Is All You Need".

    Given a batch of token sequences shaped (T, B, d_model), adds a fixed
    positional signal to each timestep so the Transformer can tell positions
    apart. Stored as a buffer so it moves with .to(device) but isn't a learned
    parameter.

    For MID we use d_model=2*context_dim (=512 when encoder_dim=256) and
    max_len=24 — well above the 12 future timesteps we ever need.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe[pos, 2i]   = sin(pos / 10000^(2i/d_model))
        # pe[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, d_model)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class ConcatSquashLinear(nn.Module):
    """
    Context-conditioned linear layer used throughout the diffusion network.

    Given input x (shape ..., dim_in) and a context vector ctx (shape ..., dim_ctx)
    that encodes (diffusion timestep β, encoder latent f), computes:

        out = Linear_in(x) * sigmoid(Linear_gate(ctx)) + Linear_bias(ctx)

    The gate (multiplicative) and bias (additive) are functions of the context,
    so every layer's behavior is modulated by both the noise level and the
    pedestrian's encoded history. This is a FiLM-style conditioning trick —
    cheap, expressive, no normalization layers needed.
    """

    def __init__(self, dim_in: int, dim_out: int, dim_ctx: int):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)
        # bias=False: Linear_bias provides an additive shift; an extra bias
        # term would be redundant
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)

    def forward(self, ctx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        return self._layer(x) * gate + bias
