"""
cvae_autoencoder.py — Trajectron++ encoder + CVAE decoder (the ablation).

Same encoder, different decoder. The diffusion version lives in autoencoder.py.

This is the model used in the MID paper's Table 3 group-4 ablation: a CVAE
decoder swapped in for the diffusion decoder while holding the encoder fixed.
We deliberately keep this drop-in compatible with the same training loop and
the same eval functions (`mid_model.eval.evaluate`) — only `get_loss` and
`generate` differ internally.
"""

import torch
import torch.nn as nn

from .cvae import CVAETraj


class CVAEAutoEncoder(nn.Module):
    """
    Trajectron++ encoder + CVAE decoder.

    The training loop, eval loop, and dataloader are unchanged from the
    diffusion model. The only public differences vs. AutoEncoder:
        - .diffusion is replaced by .cvae (a CVAETraj instance)
        - constructor takes CVAE-specific hyperparams (z_dim, kl_weight)
          instead of diffusion-specific ones (num_diffusion_steps, β schedule)
    """

    def __init__(
        self,
        encoder,
        registrar: nn.Module,
        encoder_dim: int = 256,
        z_dim: int = 32,
        future_hidden: int = 128,
        decoder_hidden: int = 256,
        future_steps: int = 12,
        kl_weight: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.registrar = registrar
        self.add_module("registrar", registrar)

        self.cvae = CVAETraj(
            context_dim=encoder_dim,
            z_dim=z_dim,
            future_hidden=future_hidden,
            decoder_hidden=decoder_hidden,
            future_steps=future_steps,
            point_dim=2,
            kl_weight=kl_weight,
        )

    def encode(self, batch, node_type) -> torch.Tensor:
        return self.encoder.get_latent(batch, node_type)

    def get_loss(self, batch, node_type) -> torch.Tensor:
        """ELBO: reconstruction MSE + β · KL[q ‖ p]."""
        (_first_history_index,
         _x_t, y_t, _x_st_t, _y_st_t,
         _neighbors_data_st,
         _neighbors_edge_value,
         _robot_traj_st_t,
         _map) = batch

        device = next(self.parameters()).device
        feat = self.encode(batch, node_type)
        return self.cvae.get_loss(y_t.to(device), feat)

    @torch.no_grad()
    def generate(
        self,
        batch,
        node_type,
        num_points: int = 12,
        sample: int = 20,
        bestof: bool = True,
        flexibility: float = 0.0,    # ignored — diffusion-only knob
        sampling: str = "ddpm",      # ignored
        step: int = 1,               # ignored
    ) -> torch.Tensor:
        """
        Sample futures and integrate to positions. Same return contract as
        AutoEncoder.generate so the existing evaluate() loop works unchanged.

        Returns: (sample, B, T, 2) in absolute scene coordinates.
        """
        dynamics = self.encoder.node_models_dict[node_type].dynamic
        feat = self.encode(batch, node_type)
        predicted_vel = self.cvae.sample(
            num_points=num_points,
            context=feat,
            sample=sample,
            bestof=bestof,
        )
        return dynamics.integrate_samples(predicted_vel)
