"""Provided A3 infrastructure helpers; graded recognition logic is not implemented here."""

from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from cs5243.data import find_course_root, find_repository_root


def assignment_paths() -> dict[str, Path]:
    root = find_repository_root(__file__); assignment = find_course_root(__file__) / "A3"
    return {"root": root, "assignment": assignment, "assets": assignment / "data",
            "figures": assignment / "outputs" / "figures", "tables": assignment / "outputs" / "tables",
            "models": assignment / "outputs" / "models", "extension": assignment / "outputs" / "extension"}


def ensure_output_directories() -> dict[str, Path]:
    paths = assignment_paths()
    for key in ("figures", "tables", "models", "extension"): paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def verify_asset_hashes() -> list[str]:
    root = assignment_paths()["assets"]; problems = []
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1); path = root / relative.strip()
        if not path.is_file(): problems.append(f"Missing asset: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected: problems.append(f"Checksum mismatch: {relative}")
    return problems


def dataset_path() -> Path:
    return assignment_paths()["assets"] / "dataset" / "visual_patterns.npz"


def dataset_metadata() -> dict:
    return json.loads((assignment_paths()["assets"] / "metadata" / "dataset.json").read_text())


def weight_path(architecture: str) -> Path:
    names = {"mobilenet_v3_small": "mobilenet_v3_small_imagenet.weights",
             "efficientnet_b0": "efficientnet_b0_imagenet.weights"}
    if architecture not in names: raise ValueError(f"Unsupported architecture: {architecture}")
    return assignment_paths()["assets"] / "models" / names[architecture]


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def available_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def save_state_dict(path: str | Path, model: torch.nn.Module) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target); return target


SKIN_CANCER_SPLIT_SEED = 5243


def resplit_by_lesion(raw: "datasets.DatasetDict") -> "datasets.DatasetDict":
    """Build lesion-level train/validation/test splits for marmal88/skin_cancer.

    Pools the loaded splits, keeps one row per ``image_id``, and assigns whole lesions to
    train/validation/test (about 71/14/14) with class stratification and a fixed course
    seed, so every image of a lesion stays in one split and every student gets the same
    splits. No network access.
    """
    import datasets
    from sklearn.model_selection import StratifiedGroupKFold

    pooled = datasets.concatenate_datasets([raw[name] for name in ("train", "validation", "test")])
    rows = pooled.select_columns(["image_id", "lesion_id", "dx"]).to_pandas()
    rows = rows.drop_duplicates("image_id").sort_values("image_id")
    folds = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SKIN_CANCER_SPLIT_SEED)
    fold_of = {}
    for fold, (_, members) in enumerate(folds.split(rows, rows["dx"], rows["lesion_id"])):
        fold_of.update({index: fold for index in rows.index[members]})
    split_of = lambda fold: "test" if fold == 0 else "validation" if fold == 1 else "train"
    indices = {"train": [], "validation": [], "test": []}
    for index in rows.index:
        indices[split_of(fold_of[index])].append(int(index))
    return datasets.DatasetDict({name: pooled.select(members) for name, members in indices.items()})
