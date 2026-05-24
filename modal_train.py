"""
Run MID training on Modal — LOO cross-validation across ETH/UCY scenes.

Mirrors notebooks/train.ipynb. Each fold is an independent model:
  - train on the merged 4 non-held-out scenes
  - track val loss on <held_out>_val.pkl per epoch
  - at the end, Best-of-K ADE/FDE on <held_out>_test.pkl
  - save mid_loo_<held_out>.pt

One-time setup:
    pip install modal && modal setup
    modal volume create mid-data
    modal volume put mid-data ./processed_data /processed_data

Train one fold:
    modal run modal_train.py                       # eth held out
    modal run modal_train.py --single-fold hotel
    modal run modal_train.py --epochs 2            # smoke test

Train all 5 folds IN PARALLEL (5 containers, ~one fold's wall time):
    modal run modal_train.py --all-folds

Download checkpoints when done:
    modal volume get mid-data /checkpoints ./checkpoints_modal
"""

import modal

app = modal.App("mid-train")

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
    cpu=8.0,
    volumes={"/data": volume},
    timeout=60 * 60 * 4,  # 4h per fold; ~2.25h expected
)
def train_fold(
    held_out: str,
    epochs: int = 90,
    batch_size: int = 256,
    lr: float = 1e-3,
    encoder_dim: int = 256,
    tf_layer: int = 3,
    num_diffusion_steps: int = 100,
    beta_1: float = 1e-4,
    beta_T: float = 5e-2,
    augment: bool = True,
    num_samples: int = 20,
    seed: int = 123,
    num_workers: int = 4,
):
    import os
    import sys
    import time
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
        f"[{held_out}] Device: {device}  ({torch.cuda.get_device_name(0)})",
        flush=True,
    )

    from mid_model import (
        AutoEncoder,
        build_dataloader,
        evaluate,
        get_hyperparameters,
        iter_loo_folds,
        load_environment,
    )
    from models.trajectron import Trajectron
    from utils.model_registrar import ModelRegistrar

    def _pick_random_augmentation(scene):
        scene_aug = np.random.choice(scene.augmented)
        scene_aug.temporal_scene_graph = scene.temporal_scene_graph
        return scene_aug

    def patch_aug_func(env):
        for s in env.scenes:
            if getattr(s, "aug_func", None) is not None:
                s.aug_func = _pick_random_augmentation

    ckpt_dir = "/data/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # iter_loo_folds with single_fold yields exactly one fold.
    held_out_, train_env, test_env = next(
        iter_loo_folds("/data/processed_data", single_fold=held_out)
    )
    assert held_out_ == held_out

    print(
        f"[{held_out}] train scenes: {len(train_env.scenes)}  "
        f"|  test scenes: {len(test_env.scenes)}",
        flush=True,
    )
    patch_aug_func(train_env)

    hyperparams = get_hyperparameters(encoder_dim=encoder_dim)
    hyperparams["batch_size"] = batch_size

    train_loader, node_type = build_dataloader(
        env=train_env,
        hyperparams=hyperparams,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        augment=augment,
    )
    test_loader, _ = build_dataloader(
        env=test_env,
        hyperparams=hyperparams,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        augment=False,
    )
    val_env = load_environment(f"/data/processed_data/{held_out}_val.pkl")
    patch_aug_func(val_env)
    val_loader, _ = build_dataloader(
        env=val_env,
        hyperparams=hyperparams,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        augment=False,
    )
    print(
        f"[{held_out}] batches/epoch: {len(train_loader)}  "
        f"|  val: {len(val_loader)}  |  test: {len(test_loader)}",
        flush=True,
    )

    registrar = ModelRegistrar(model_dir=ckpt_dir, device=device)
    encoder = Trajectron(registrar, hyperparams, device)
    encoder.set_environment(train_env)
    encoder.set_annealing_params()

    model = AutoEncoder(
        encoder=encoder,
        registrar=registrar,
        encoder_dim=encoder_dim,
        num_diffusion_steps=num_diffusion_steps,
        beta_1=beta_1,
        beta_T=beta_T,
        tf_layer=tf_layer,
    ).to(device)
    print(
        f"[{held_out}] params: {sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch = next(iter(train_loader))
    model.train()
    smoke = model.get_loss(batch, node_type)
    assert torch.isfinite(smoke), f"[{held_out}] smoke-test loss is not finite"
    print(f"[{held_out}] smoke-test loss: {smoke.item():.4f}", flush=True)

    history = []
    val_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        ep_losses = []
        t0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.get_loss(batch, node_type)
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())
        train_loss = float(np.mean(ep_losses))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                val_losses.append(model.get_loss(batch, node_type).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        elapsed = time.time() - t0
        history.append(train_loss)
        val_history.append(val_loss)
        print(
            f"[{held_out}] epoch {epoch:>3}/{epochs}: "
            f"train={train_loss:.4f}  val={val_loss:.4f}  ({elapsed:.1f}s)",
            flush=True,
        )

        if epoch % 10 == 0 or epoch == epochs:
            ckpt_path = os.path.join(ckpt_dir, f"mid_loo_{held_out}.pt")
            torch.save(
                {
                    "scene": held_out,
                    "epoch": epoch,
                    "hyperparams": hyperparams,
                    "encoder_dim": encoder_dim,
                    "tf_layer": tf_layer,
                    "num_diffusion_steps": num_diffusion_steps,
                    "registrar_state_dict": registrar.model_dict.state_dict(),
                    "diffusion_state_dict": model.diffusion.state_dict(),
                    "history": history,
                    "val_history": val_history,
                },
                ckpt_path,
            )
            volume.commit()

    eth_rescale = held_out == "eth"
    results = evaluate(
        model=model,
        dataloader=test_loader,
        node_type=node_type,
        device=device,
        sample=num_samples,
        eth_rescale=eth_rescale,
    )
    ade = float(results["ade"])
    fde = float(results["fde"])
    print(
        f"[{held_out}] Best-of-{num_samples}  ADE={ade:.4f}  FDE={fde:.4f}",
        flush=True,
    )

    ckpt_path = os.path.join(ckpt_dir, f"mid_loo_{held_out}.pt")
    torch.save(
        {
            "scene": held_out,
            "epoch": epochs,
            "hyperparams": hyperparams,
            "encoder_dim": encoder_dim,
            "tf_layer": tf_layer,
            "num_diffusion_steps": num_diffusion_steps,
            "registrar_state_dict": registrar.model_dict.state_dict(),
            "diffusion_state_dict": model.diffusion.state_dict(),
            "history": history,
            "val_history": val_history,
            "ade": ade,
            "fde": fde,
        },
        ckpt_path,
    )
    volume.commit()
    print(f"[{held_out}] saved: {ckpt_path}", flush=True)

    return {"held_out": held_out, "ade": ade, "fde": fde}


@app.local_entrypoint()
def main(
    single_fold: str = "eth",
    all_folds: bool = False,
    epochs: int = 90,
    batch_size: int = 256,
    num_workers: int = 4,
):
    common = dict(epochs=epochs, batch_size=batch_size, num_workers=num_workers)

    if all_folds:
        print(f"Fanning out {len(ALL_FOLDS)} folds in parallel: {ALL_FOLDS}")
        # spawn returns a FunctionCall handle; .get() blocks until that fold finishes.
        # All 5 spawns kick off immediately, so total wall time ≈ slowest fold.
        handles = [train_fold.spawn(f, **common) for f in ALL_FOLDS]
        results = [h.get() for h in handles]
    else:
        results = [train_fold.remote(single_fold, **common)]

    import statistics

    print("\n────────────────────────────────────────")
    print("    Fold    ADE      FDE")
    print("────────────────────────────────────────")
    for r in results:
        print(f"    {r['held_out']:<7} {r['ade']:<8.4f} {r['fde']:<8.4f}")
    if len(results) > 1:
        mean_ade = statistics.mean(r["ade"] for r in results)
        mean_fde = statistics.mean(r["fde"] for r in results)
        print("────────────────────────────────────────")
        print(f"    {'mean':<7} {mean_ade:<8.4f} {mean_fde:<8.4f}")
    print("────────────────────────────────────────")
