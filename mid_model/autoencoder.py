"""
autoencoder.py — wires the Trajectron++ encoder to our diffusion noise predictor.

This is the model that gets trained end-to-end. It owns:
    - self.encoder  : imported Trajectron object from MID/models/encoders/.
                      Produces the latent f (shape B, encoder_dim) from history
                      + neighbor trajectories.
    - self.diffusion: our DiffusionTraj (mid_model/diffusion.py), which takes f
                      as context and learns to denoise the future trajectory.

`get_loss(batch, node_type)` is the only method the training loop calls.

Differences vs MID's AutoEncoder:
    - Device-agnostic: no `.cuda()` calls. The model picks up the device
      of its parameters via the trick `next(self.parameters()).device`.
    - Encoder is registered as a non-Module attribute (it's not a torch.nn.Module
      — `Trajectron` is a plain object that wraps a ModelRegistrar). We expose
      `self.registrar` for the optimizer to grab its parameters.
"""

from typing import Optional

import torch
import torch.nn as nn

from .diffusion import DiffusionTraj, TransformerConcatLinear, VarianceSchedule


class AutoEncoder(nn.Module):
    """
    Combines the Trajectron++ encoder with our diffusion decoder.

    Args:
        encoder: an already-constructed Trajectron object (MID class). Must
                 have had `.set_environment(env)` and `.set_annealing_params()`
                 called before passing in.
        registrar: the encoder's ModelRegistrar — needed so we can register
                   its parameters with the optimizer. (Trajectron itself is
                   not an nn.Module; its sub-models live in this registrar.)
        encoder_dim: dimensionality of the latent f. Must match the encoder's
                     enc_rnn_dim_history + enc_rnn_dim_edge_influence (= 2*half
                     when set up via hypers.get_hyperparameters(encoder_dim)).
        num_diffusion_steps: K in the DDPM process. Paper uses 100.
        beta_1, beta_T: linear schedule endpoints.
        tf_layer: number of Transformer encoder layers in the noise net.
    """

    def __init__(
        self,
        encoder,
        registrar: nn.Module,
        encoder_dim: int = 256,
        num_diffusion_steps: int = 100,
        beta_1: float = 1e-4,
        beta_T: float = 5e-2,
        tf_layer: int = 3,
    ):
        super().__init__()
        self.encoder = encoder            # not registered as a submodule on purpose
        self.registrar = registrar        # IS an nn.Module; its params show up in self.parameters()
        self.add_module("registrar", registrar)  # explicit so optimizer collects them

        self.diffusion = DiffusionTraj(
            net=TransformerConcatLinear(
                point_dim=2,
                context_dim=encoder_dim,
                tf_layer=tf_layer,
                residual=False,
            ),
            var_sched=VarianceSchedule(
                num_steps=num_diffusion_steps,
                beta_1=beta_1,
                beta_T=beta_T,
                mode="linear",
            ),
        )

    # ── Forward paths ──

    def encode(self, batch, node_type) -> torch.Tensor:
        """Run only the encoder. Returns f of shape (B, encoder_dim)."""
        return self.encoder.get_latent(batch, node_type)

    def get_loss(self, batch, node_type) -> torch.Tensor:
        """
        DDPM ε-prediction loss on the future trajectory.

        The encoder produces context f from history. The diffusion model
        denoises y_t (future positions in the *raw absolute frame*, shape
        (B, 12, 2)) conditioned on f.

        Note: we use y_t, NOT y_st_t. The diffusion model operates in the raw
        coordinate space; the encoder consumes the standardized inputs
        internally. This matches MID's setup.
        """
        # Unpack just to grab y_t; the encoder will use the full batch tuple.
        (_first_history_index,
         _x_t, y_t, _x_st_t, _y_st_t,
         _neighbors_data_st,
         _neighbors_edge_value,
         _robot_traj_st_t,
         _map) = batch

        device = next(self.parameters()).device
        feat = self.encode(batch, node_type)          # (B, encoder_dim), already on encoder.device
        return self.diffusion.get_loss(y_t.to(device), feat)

    @torch.no_grad()
    def generate(
        self,
        batch,
        node_type,
        num_points: int = 12,
        sample: int = 20,
        bestof: bool = True,
        flexibility: float = 0.0,
        sampling: str = "ddpm",
        step: int = 1,
    ) -> torch.Tensor:
        """
        Sample future trajectories. For Best-of-20 evaluation, set sample=20.

        Returns predicted position trajectories of shape (sample, B, 12, 2),
        obtained by:
            1. Encoding the batch to get context f.
            2. Running the reverse diffusion chain to sample velocities.
            3. Integrating velocities through the dynamics model (cumulative
               sum for SingleIntegrator) to get positions.
        """
        dynamics = self.encoder.node_models_dict[node_type].dynamic
        feat = self.encode(batch, node_type)
        predicted_vel = self.diffusion.sample(
            num_points=num_points,
            context=feat,
            sample=sample,
            bestof=bestof,
            flexibility=flexibility,
            sampling=sampling,
            step=step,
        )
        # (sample, B, T, 2)
        predicted_pos = dynamics.integrate_samples(predicted_vel)
        return predicted_pos
