"""
Run MID evaluation (Best-of-K ADE/FDE) on Modal.

Assumes you've already trained on Modal — LOO checkpoints live in the
shared `mid-data` volume at `/data/checkpoints/mid_loo_<scene>.pt`.

Each fold's checkpoint is evaluated on that fold's held-out *test* split
(or *val* if you pass --split val).

Eval one fold:
    modal run modal_eval.py                          # eth / test, K=20, DDPM full chain
    modal run modal_eval.py --scene hotel
    modal run modal_eval.py --scene eth --split val
    modal run modal_eval.py --sampling ddim --step 10   # fast sampling

Eval all 5 folds in parallel:
    modal run modal_eval.py --all-folds
    modal run modal_eval.py --all-folds --sampling ddim --step 10

Pull per-pedestrian arrays back for plotting:
    modal volume get mid-data /eval_results ./eval_results
"""

import modal

app = modal.App("mid-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "numpy<2",
        "pandas",
        "scipy",
        "scikit-learn",
        "tqdm",
        "dill",
        "h5py",
        "ncls",
        "orjson",
        "pyyaml",
        "easydict",
        "pyquaternion",
    )
    .add_local_dir("./mid_model", "/root/mid_model")
    .add_local_dir("./models", "/root/models")
    .add_local_dir("./environment", "/root/environment")
    .add_local_dir("./utils", "/root/utils")
    .add_local_dir("./dataset", "/root/dataset")
)

volume = modal.Volume.from_name("mid-data", create_if_missing=True)

ALL_FOLDS = ["eth", "hotel", "univ", "zara1", "zara2"]


@app.function(
    image=image,
    gpu="T4",
    cpu=4.0,
    volumes={"/data": volume},
    timeout=60 * 60 * 2,
)
def evaluate_scene(
    scene: str = "eth",
    split: str = "test",
    batch_size: int = 256,
    num_samples: int = 20,
    sampling: str = "ddpm",
    step: int = 1,
    min_history: int = 7,
    seed: int = 123,
):
    import os
    import sys
    import random

    import numpy as np
    import torch

    sys.path.insert(0, "/root")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    print(
        f"[{scene}/{split}] Device: {device}  ({torch.cuda.get_device_name(0)})",
        flush=True,
    )

    from mid_model import (
        AutoEncoder,
        build_dataloader,
        evaluate,
        get_hyperparameters,
        load_environment,
    )
    from models.trajectron import Trajectron
    from utils.model_registrar import ModelRegistrar

    ckpt_path = f"/data/checkpoints/mid_loo_{scene}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    print(
        f"[{scene}/{split}] Loaded {ckpt_path}\n"
        f"  trained on scene: {ckpt['scene']}\n"
        f"  epochs:           {ckpt['epoch']}\n"
        f"  encoder_dim:      {ckpt['encoder_dim']}\n"
        f"  diffusion steps:  {ckpt['num_diffusion_steps']}",
        flush=True,
    )
    if ckpt.get("history"):
        print(f"  final train loss: {ckpt['history'][-1]:.4f}", flush=True)
    if ckpt.get("val_history"):
        print(f"  final val loss:   {ckpt['val_history'][-1]:.4f}", flush=True)
    if "ade" in ckpt:
        print(
            f"  ade@train-end:    {ckpt['ade']:.4f}   fde@train-end: {ckpt['fde']:.4f}",
            flush=True,
        )

    pkl_path = f"/data/processed_data/{scene}_{split}.pkl"
    env = load_environment(pkl_path)
    print(
        f"[{scene}/{split}] Loaded {pkl_path}  scenes={len(env.scenes)}  "
        f"nodes={sum(len(s.nodes) for s in env.scenes)}",
        flush=True,
    )

    hyperparams = ckpt.get("hyperparams") or get_hyperparameters(
        encoder_dim=ckpt["encoder_dim"]
    )
    hyperparams = dict(hyperparams)
    hyperparams["minimum_history_length"] = min_history

    eval_loader, node_type = build_dataloader(
        env=env,
        hyperparams=hyperparams,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        augment=False,
    )
    print(
        f"[{scene}/{split}] node type: {node_type}  batches: {len(eval_loader)}",
        flush=True,
    )

    registrar = ModelRegistrar(model_dir="/data/checkpoints", device=device)
    encoder = Trajectron(registrar, hyperparams, device)
    encoder.set_environment(env)
    encoder.set_annealing_params()

    model = AutoEncoder(
        encoder=encoder,
        registrar=registrar,
        encoder_dim=ckpt["encoder_dim"],
        num_diffusion_steps=ckpt["num_diffusion_steps"],
        beta_1=1e-4,
        beta_T=5e-2,
        tf_layer=ckpt["tf_layer"],
    ).to(device)

    registrar.model_dict.load_state_dict(ckpt["registrar_state_dict"])
    model.diffusion.load_state_dict(ckpt["diffusion_state_dict"])
    print(f"[{scene}/{split}] Weights loaded.", flush=True)

    eth_rescale = scene == "eth" and split == "test"

    results = evaluate(
        model=model,
        dataloader=eval_loader,
        node_type=node_type,
        device=device,
        sample=num_samples,
        sampling=sampling,
        step=step,
        eth_rescale=eth_rescale,
    )

    print(f"\n══════ {scene} / {split} ═══════════════════", flush=True)
    print(f"  Sampling:        {sampling} (step={step})", flush=True)
    print(f"  Pedestrians:     {results['n_pedestrians']}", flush=True)
    print(f"  Best-of-{num_samples} ADE: {results['ade']:.4f}", flush=True)
    print(f"  Best-of-{num_samples} FDE: {results['fde']:.4f}", flush=True)
    if eth_rescale:
        print("  (ADE/FDE rescaled by 1/0.6 for the ETH benchmark frame)", flush=True)
    print("══════════════════════════════════════════════", flush=True)

    out_dir = "/data/eval_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"{scene}_{split}_{sampling}_step{step}_K{num_samples}.npz"
    )
    np.savez(
        out_path,
        ade=results["ade"],
        fde=results["fde"],
        ade_per_pedestrian=results["ade_per_pedestrian"],
        fde_per_pedestrian=results["fde_per_pedestrian"],
        n_pedestrians=results["n_pedestrians"],
    )
    volume.commit()
    print(f"[{scene}/{split}] saved per-pedestrian arrays to {out_path}", flush=True)

    return {
        "scene": scene,
        "split": split,
        "ade": float(results["ade"]),
        "fde": float(results["fde"]),
        "n_pedestrians": int(results["n_pedestrians"]),
    }


@app.local_entrypoint()
def main(
    scene: str = "eth",
    all_folds: bool = False,
    split: str = "test",
    num_samples: int = 20,
    sampling: str = "ddpm",
    step: int = 1,
    batch_size: int = 256,
    min_history: int = 7,
):
    common = dict(
        split=split,
        batch_size=batch_size,
        num_samples=num_samples,
        sampling=sampling,
        step=step,
        min_history=min_history,
    )

    if all_folds:
        print(f"Fanning out {len(ALL_FOLDS)} eval folds in parallel: {ALL_FOLDS}")
        handles = [evaluate_scene.spawn(s, **common) for s in ALL_FOLDS]
        results = [h.get() for h in handles]
    else:
        results = [evaluate_scene.remote(scene, **common)]

    import statistics

    print("\n────────────────────────────────────────")
    print(f"    Scene   ADE      FDE      (K={num_samples}, {sampling}, step={step})")
    print("────────────────────────────────────────")
    for r in results:
        print(f"    {r['scene']:<7} {r['ade']:<8.4f} {r['fde']:<8.4f}")
    if len(results) > 1:
        mean_ade = statistics.mean(r["ade"] for r in results)
        mean_fde = statistics.mean(r["fde"] for r in results)
        print("────────────────────────────────────────")
        print(f"    {'mean':<7} {mean_ade:<8.4f} {mean_fde:<8.4f}")
    print("────────────────────────────────────────")
