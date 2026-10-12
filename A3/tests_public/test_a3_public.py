"""Limited public sanity checks for A3."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import a3_tools  # noqa: E402

candidate = os.environ.get("CS5243_A3_IMPLEMENTATION")
if candidate:
    spec = importlib.util.spec_from_file_location("a3_candidate", candidate)
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
else:
    import student_code as sc  # noqa: E402

NAMES = [
    "build_dataloaders",
    "build_model",
    "freeze_backbone",
    "train_one_epoch",
    "evaluate_model",
    "predict",
    "extract_embeddings",
    "classification_metrics",
    "plot_confusion_matrix",
    "build_external_dataloaders",
]


def run(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{function.__name__} remains a starter placeholder")


def test_imports_assets_and_signatures():
    assert all(callable(getattr(sc, name, None)) for name in NAMES)
    assert a3_tools.verify_asset_hashes() == []


def test_dataloaders_are_reproducible():
    first = run(sc.build_dataloaders, a3_tools.dataset_path(), 8, "none", 17, 0)
    second = run(sc.build_dataloaders, a3_tools.dataset_path(), 8, "none", 17, 0)
    assert set(first) == {"train", "validation", "test"}
    batch_a = next(iter(first["train"]))
    batch_b = next(iter(second["train"]))
    assert torch.equal(batch_a[0], batch_b[0])
    assert torch.equal(batch_a[1], batch_b[1])


def test_model_and_freezing_contract():
    model = run(
        sc.build_model,
        "mobilenet_v3_small",
        6,
        a3_tools.weight_path("mobilenet_v3_small"),
    )
    assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 6)
    run(sc.freeze_backbone, model)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all("classifier" in name for name in trainable)


def test_prediction_metrics_and_embeddings():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Flatten()
            self.classifier = nn.Sequential(
                nn.Linear(12, 5),
                nn.ReLU(),
                nn.Linear(5, 3),
            )

        def forward(self, inputs):
            return self.classifier(self.features(inputs))

    model = Tiny()
    images = torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2) / 48
    labels = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    result = run(sc.predict, model, images, torch.device("cpu"))
    assert result["predictions"].shape == (4,)
    assert result["probabilities"].shape == (4, 3)

    embeddings, out_labels = run(
        sc.extract_embeddings,
        model,
        loader,
        torch.device("cpu"),
    )
    assert embeddings.shape == (4, 5)
    assert np.array_equal(out_labels, labels.numpy())

    metrics = run(
        sc.classification_metrics,
        labels.numpy(),
        np.array([0, 2, 2, 1]),
        ["a", "b", "c"],
    )
    assert metrics["confusion_matrix"].shape == (3, 3)
    assert 0 <= metrics["accuracy"] <= 1
