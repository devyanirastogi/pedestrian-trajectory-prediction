"""
dataset.py — wrap an Environment .pkl into a PyTorch DataLoader.

We REUSE MID's dataset/dataset.py + dataset/preprocessing.py here, because:
    1. `EnvironmentDataset` indexes (scene, timestep, node) tuples by walking
       the Environment object and calling `scene.present_nodes(...)`. Faithful
       to the data model.
    2. `get_node_timestep_data` is the workhorse that pulls (x_t, y_t, x_st_t,
       y_st_t, neighbors, ...) out of the Environment — it's tightly coupled
       to the Node API and is correct as written.
    3. The custom `collate` handles variable-length neighbor lists by leaving
       them as Python lists rather than padding to a tensor; the encoder
       expects exactly that shape.

So this file is mostly glue: load the .pkl, set hyperparams correctly,
construct the DataLoader. The notebook calls `build_dataloader(...)`.
"""

import os
import sys
import dill
import torch
from torch.utils.data import DataLoader

# Ensure PROJECT_ROOT is on sys.path so `dataset.dataset`, `dataset.preprocessing`,
# and `environment.*` (used by dill at load time) all resolve to the local copies
# we keep at the project root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset.dataset import EnvironmentDataset  # noqa: E402
from dataset.preprocessing import collate        # noqa: E402


def load_environment(pkl_path: str):
    """
    Load a preprocessed Environment from disk.

    The .pkl was written by preprocess_ethucy.py via dill, so we use dill
    (not pickle) and set encoding='latin1' for cross-version compatibility.

    Gotcha — stale dill-serialized classes:
      Dill embeds class bytecode directly into the pickle, not just a module
      reference. If the .pkl was created BEFORE we patched scene_graph.py
      (np.float → float), the loaded scenes carry the old class object inside
      their method __globals__, which shadows our patched module. The fix
      below re-injects the patched TemporalSceneGraph / SceneGraph classes
      into each scene's globals so the live patched code runs.
      Re-running preprocess.ipynb is the proper long-term fix; this keeps
      old .pkl files working.
    """
    with open(pkl_path, "rb") as f:
        env = dill.load(f, encoding="latin1")

    # Sanity check: the Environment should already have scenes and
    # an attention_radius dict set up by the preprocessing step.
    assert hasattr(env, "scenes") and len(env.scenes) > 0, \
        f"Environment in {pkl_path} has no scenes"
    assert hasattr(env, "attention_radius"), \
        f"Environment in {pkl_path} missing attention_radius"

    # Stale-class workaround (see docstring).
    from environment.scene_graph import TemporalSceneGraph, SceneGraph
    for scene in env.scenes:
        g = scene.get_scene_graph.__globals__
        g["TemporalSceneGraph"] = TemporalSceneGraph
        g["SceneGraph"] = SceneGraph

    return env


def build_dataloader(
    env,
    hyperparams: dict,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    augment: bool = False,
):
    """
    Build a PyTorch DataLoader for a single node type (PEDESTRIAN).

    EnvironmentDataset internally creates one NodeTypeDataset per node type
    listed in `hyperparams['pred_state']`. For ETH/UCY that's just PEDESTRIAN,
    so we unwrap and return a single DataLoader rather than a dict.

    Args:
        env: Environment loaded via `load_environment`.
        hyperparams: dict from `get_hyperparameters()`.
        batch_size: training batch size (256 per the paper).
        shuffle: True for train, False for val/test.
        num_workers: 0 is fine for ETH/UCY (small dataset, the preprocessing
            inside __getitem__ is cheap). >0 triggers dill-encoding of the
            neighbor dicts inside collate, which adds overhead.
        augment: whether to apply random rotation augmentation (training only).
            Requires `scene.augmented` to be populated, which the preprocessing
            does for `train` splits.

    Returns:
        (dataloader, node_type)
            dataloader: torch.utils.data.DataLoader
            node_type:  the env.NodeType.PEDESTRIAN enum value, needed when
                        calling encoder.get_latent(batch, node_type)
    """
    dataset = EnvironmentDataset(
        env=env,
        state=hyperparams["state"],
        pred_state=hyperparams["pred_state"],
        node_freq_mult=hyperparams["node_freq_mult_train"],
        scene_freq_mult=hyperparams["scene_freq_mult_train"],
        hyperparams=hyperparams,
        min_history_timesteps=hyperparams["minimum_history_length"],
        min_future_timesteps=hyperparams["prediction_horizon"],
        return_robot=not hyperparams["incl_robot_node"],
    )
    dataset.augment = augment

    # EnvironmentDataset is iterable over per-node-type sub-datasets.
    # For PEDESTRIAN-only data, this yields exactly one.
    node_type_datasets = list(dataset)
    assert len(node_type_datasets) == 1, \
        f"expected 1 node type (PEDESTRIAN), got {len(node_type_datasets)}"
    node_type_dataset = node_type_datasets[0]

    loader = DataLoader(
        node_type_dataset,
        collate_fn=collate,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, node_type_dataset.node_type
