"""
eval.py — ADE/FDE Best-of-K evaluation for the diffusion trajectory model.

Mirrors MID's evaluation semantics:

    ADE = mean_t || pred[:, t] - gt[:, t] ||_2           (per pedestrian, per sample)
    FDE = || pred[:, T-1] - gt[:, T-1] ||_2              (per pedestrian, per sample)
    Best-of-K → min across K samples per pedestrian
    Aggregate → mean across all pedestrians

Both predictions and ground truth are in absolute scene coordinates. The
diffusion model outputs velocities; the encoder's dynamics module integrates
them to positions using the agent's current position as p₀ (set inside the
encoder's forward pass). We integrate the ground-truth velocity y_t with the
SAME dynamics call so both pred and gt share an anchor — any small drift
from Euler-vs-gradient integration cancels.

A note on the ETH 0.6× quirk: MID applies it post-hoc to the final ADE/FDE.
Our preprocessing already scaled the ETH test set's coordinates by 0.6, so
*if* you evaluate the eth_test.pkl you must rescale (divide) the result by 0.6
to match the literature's reporting frame. Toggle via `eth_rescale=True`.
"""

from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm


# ─────────────────────────────────────────────────────────────────────
# Error metrics (work on numpy arrays for speed/clarity)
# ─────────────────────────────────────────────────────────────────────
def compute_ade(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Average displacement error per (sample, pedestrian).

    pred: (S, B, T, 2) — K=S samples of predicted positions
    gt:   (B, T, 2)    — ground-truth positions, broadcast across S
    returns: (S, B)
    """
    err = np.linalg.norm(pred - gt[None], axis=-1)  # (S, B, T)
    return err.mean(axis=-1)                          # (S, B)


def compute_fde(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Final displacement error per (sample, pedestrian). Shape (S, B)."""
    err = np.linalg.norm(pred[..., -1, :] - gt[None, ..., -1, :], axis=-1)
    return err


# ─────────────────────────────────────────────────────────────────────
# End-to-end evaluation loop
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model,
    dataloader,
    node_type,
    device: torch.device,
    sample: int = 20,
    sampling: str = "ddpm",
    step: int = 1,
    eth_rescale: bool = False,
    progress: bool = True,
) -> dict:
    """
    Run Best-of-K ADE/FDE evaluation over an entire DataLoader.

    Args:
        model:       AutoEncoder (must already be on `device` with weights loaded).
        dataloader:  yields batches in the same 9-tuple format as training.
        node_type:   env.NodeType.PEDESTRIAN
        device:      torch device — must match model's device.
        sample:      K in Best-of-K. Paper reports K=20.
        sampling:    "ddpm" (paper default) or "ddim" (faster).
        step:        reverse-process stride. 1 = full K diffusion steps.
        eth_rescale: divide final metrics by 0.6 (only meaningful on eth_test).
        progress:    show tqdm bar.

    Returns dict with:
        ade:  scalar — mean over pedestrians of min-over-samples ADE
        fde:  scalar — same for FDE
        ade_per_pedestrian: 1D array length N_pedestrians
        fde_per_pedestrian: 1D array length N_pedestrians
        n_pedestrians: total pedestrians evaluated
    """
    model.eval()

    ade_chunks, fde_chunks = [], []
    iterator = tqdm(dataloader, desc="evaluating", ncols=100) if progress else dataloader

    for batch in iterator:
        # Run the full pipeline:
        #   encode → diffusion.sample(K times) → integrate → positions
        # pred_pos shape: (S, B, T, 2). The encoder's forward pass during
        # generate() sets `dynamics.initial_conditions`, which we reuse below
        # to integrate the ground-truth velocity for an apples-to-apples
        # comparison.
        pred_pos = model.generate(
            batch,
            node_type,
            num_points=12,
            sample=sample,
            bestof=True,
            sampling=sampling,
            step=step,
        )

        # Ground truth: integrate y_t (future velocity) with the dynamics
        # module whose initial condition was just set by the encoder pass.
        (_first_history_index,
         _x_t, y_t, _x_st_t, _y_st_t,
         _neighbors_data_st,
         _neighbors_edge_value,
         _robot_traj_st_t,
         _map) = batch

        dynamics = model.encoder.node_models_dict[node_type].dynamic
        y_t_dev = y_t.to(device).unsqueeze(0)               # (1, B, T, 2)
        gt_pos = dynamics.integrate_samples(y_t_dev)        # (1, B, T, 2)
        gt_pos = gt_pos.squeeze(0)                          # (B, T, 2)

        pred_np = pred_pos.detach().cpu().numpy()           # (S, B, T, 2)
        gt_np = gt_pos.detach().cpu().numpy()               # (B, T, 2)

        ade = compute_ade(pred_np, gt_np)                   # (S, B)
        fde = compute_fde(pred_np, gt_np)                   # (S, B)

        # Best-of-K reduction: min across samples → one ADE/FDE per pedestrian.
        ade_chunks.append(ade.min(axis=0))                  # (B,)
        fde_chunks.append(fde.min(axis=0))                  # (B,)

    ade_all = np.concatenate(ade_chunks)
    fde_all = np.concatenate(fde_chunks)

    mean_ade = float(ade_all.mean())
    mean_fde = float(fde_all.mean())

    if eth_rescale:
        mean_ade /= 0.6
        mean_fde /= 0.6

    return {
        "ade": mean_ade,
        "fde": mean_fde,
        "ade_per_pedestrian": ade_all,
        "fde_per_pedestrian": fde_all,
        "n_pedestrians": int(ade_all.shape[0]),
    }
