"""
loo.py — Leave-one-out cross-validation utilities for ETH/UCY.

The standard ETH/UCY protocol: for each held-out scene s ∈ {eth, hotel, univ,
zara1, zara2}, train on the OTHER four scenes (their `train` splits merged)
and report ADE/FDE on s's `test` split. So one full LOO run = 5 train/eval
cycles, each on a different fold.

This module gives you the data-side primitives. Training / eval loops live
in the notebooks; the helpers here just hand back ready-to-use Environment
objects.

Two functions:

    merge_environments(envs) → Environment
        Concatenate the `scenes` lists of several Environments into one.
        Used to build the merged training Environment from N-1 scenes.

    iter_loo_folds(processed_data_dir, all_scenes, single_fold=None)
        Yields (held_out, train_env, test_env) per fold. Set `single_fold` to
        a scene name to short-circuit and only yield that one fold.
"""

import os
from typing import Iterable, Iterator, List, Optional, Tuple


# The five ETH/UCY scenes. Importing this list saves the notebook a hardcoded copy.
ETH_UCY_SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]


def merge_environments(envs: Iterable):
    """
    Combine several Environment objects into one by concatenating their
    .scenes lists. We mutate the first env in-place (saves a deep copy of all
    the scenes) and return it. Caller should not reuse the input envs as
    independent objects after calling this.

    All envs must share the same standardization config and attention_radius
    (they do, for ETH/UCY — they all come from the same preprocessing pass).
    We assert this so a silent mismatch can't drift training.
    """
    envs = list(envs)
    if not envs:
        raise ValueError("merge_environments: need at least one env")

    merged = envs[0]
    for env in envs[1:]:
        # Sanity: the metadata that the encoder reads must agree.
        assert env.attention_radius == merged.attention_radius, \
            "attention_radius mismatch across envs"
        assert env.standardization == merged.standardization, \
            "standardization config mismatch across envs"
        merged.scenes.extend(env.scenes)
    return merged


def iter_loo_folds(
    processed_data_dir: str,
    all_scenes: Optional[List[str]] = None,
    single_fold: Optional[str] = None,
    train_split: str = "train",
    test_split: str = "test",
) -> Iterator[Tuple[str, "Environment", "Environment"]]:
    """
    Yield (held_out_scene, train_env, test_env) for each LOO fold.

    Args:
        processed_data_dir: e.g. "/Users/.../processed_data". Each scene must
            have <scene>_<split>.pkl files inside.
        all_scenes: list of scene names to fold over. Default: ETH_UCY_SCENES.
        single_fold: if set to one of the scene names, yield ONLY that fold.
            Useful during development; flip to None to run all 5.
        train_split / test_split: split names — default ("train", "test")
            matches the paper, but you can switch test_split to "val" for
            validation runs.

    Note: this re-loads .pkl files on every iteration and rebuilds the merged
    env from scratch each time. That's intentional: dill-loaded objects carry
    method `__globals__` references, and reusing them across folds would
    accumulate scenes you didn't intend.
    """
    # Import here so the module-level import doesn't pull torch et al. unless
    # someone actually uses LOO. Keeps the module light.
    from .dataset import load_environment

    scenes = list(all_scenes or ETH_UCY_SCENES)

    if single_fold is not None:
        if single_fold not in scenes:
            raise ValueError(
                f"single_fold={single_fold!r} not in scenes={scenes}"
            )
        scenes = [single_fold]

    # We always need to know the *full* scene list when picking training
    # scenes — even if single_fold is set, the training env is the OTHER four.
    full_scenes = list(all_scenes or ETH_UCY_SCENES)

    for held_out in scenes:
        train_scenes = [s for s in full_scenes if s != held_out]

        train_envs = [
            load_environment(os.path.join(processed_data_dir, f"{s}_{train_split}.pkl"))
            for s in train_scenes
        ]
        train_env = merge_environments(train_envs)

        test_env = load_environment(
            os.path.join(processed_data_dir, f"{held_out}_{test_split}.pkl")
        )

        yield held_out, train_env, test_env
