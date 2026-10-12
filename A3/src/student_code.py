"""Student implementations for A3 transfer learning and visual recognition."""

from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import a3_tools

import torchvision
from torchvision import transforms
from PIL import Image

# ImageNet statistics that match the bundled pretrained state dictionaries.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# The pretrained backbones were trained at 224x224; every split is resized to it so that
# architecture, augmentation, and optimizer comparisons differ in exactly one factor.
IMAGE_SIZE = 224
AUGMENTATION_LEVELS = ("none", "moderate", "strong")
SUPPORTED_ARCHITECTURES = ("mobilenet_v3_small", "efficientnet_b0")


def _normalization() -> list:
    return [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def evaluation_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic resize + ImageNet normalization used for validation and test."""
    return transforms.Compose([transforms.Resize((image_size, image_size))] + _normalization())


def training_transform(augmentation: str, image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Training transform for ``none``, ``moderate``, or ``strong`` augmentation."""
    if augmentation not in AUGMENTATION_LEVELS:
        raise ValueError(f"augmentation must be one of {AUGMENTATION_LEVELS}, got {augmentation!r}")
    if augmentation == "none":
        return evaluation_transform(image_size)
    if augmentation == "moderate":
        # Label-preserving for every class: flips map each pattern family onto itself.
        steps = [transforms.Resize((image_size, image_size)),
                 transforms.RandomHorizontalFlip(0.5),
                 transforms.RandomVerticalFlip(0.5),
                 transforms.ColorJitter(brightness=0.2, contrast=0.2)]
    else:
        # Deliberately aggressive: rotation and cropping can destroy orientation evidence.
        steps = [transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
                 transforms.RandomHorizontalFlip(0.5),
                 transforms.RandomVerticalFlip(0.5),
                 transforms.RandomRotation(30),
                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                 transforms.RandomGrayscale(0.1)]
    return transforms.Compose(steps + _normalization())


def _loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _drop_last(size: int, batch_size: int, shuffle: bool) -> bool:
    """Drop a trailing single-sample training batch, which batch normalization cannot use."""
    return bool(shuffle and batch_size > 1 and size % batch_size == 1)


def _worker_init(worker_id: int) -> None:  # pragma: no cover - exercised only with workers
    import random as _random
    base = torch.initial_seed() % 2**32
    np.random.seed(base)
    _random.seed(base)


class ArrayImageDataset(Dataset):
    """RGB ``uint8`` arrays (N x H x W x 3) with integer labels and a transform."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform, class_names: list[str]):
        self.images = np.ascontiguousarray(images)
        self.labels = np.asarray(labels).astype(np.int64)
        self.transform = transform
        self.class_names = list(class_names)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index]).convert("RGB")
        return self.transform(image), int(self.labels[index])


def build_dataloaders(dataset_path: str | Path, batch_size: int, augmentation: str,
                      seed: int, num_workers: int = 0) -> dict[str, DataLoader]:
    """Build deterministic train/validation/test loaders for the bundled NPZ dataset.

    The NPZ stores RGB ``uint8`` arrays ``{split}_images`` (N x 64 x 64 x 3) and integer
    ``{split}_labels`` for ``split`` in ``train``, ``validation``, and ``test``. Return a
    dictionary with keys ``train``, ``validation``, and ``test`` whose loaders yield
    ``(images, labels)`` batches.

    ``augmentation`` must be ``none``, ``moderate``, or ``strong`` and applies only
    to training. Evaluation transforms must be deterministic and use ImageNet
    normalization. Train shuffling must be controlled by ``seed``.
    """
    if augmentation not in AUGMENTATION_LEVELS:
        raise ValueError(f"augmentation must be one of {AUGMENTATION_LEVELS}, got {augmentation!r}")
    try:
        class_names = list(a3_tools.dataset_metadata()["classes"])
    except Exception:  # pragma: no cover - metadata is bundled with the assignment
        class_names = []

    archive = np.load(Path(dataset_path))
    transform_of = {"train": training_transform(augmentation),
                    "validation": evaluation_transform(), "test": evaluation_transform()}
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        labels = archive[f"{split}_labels"]
        names = class_names or [str(index) for index in range(int(labels.max()) + 1)]
        dataset = ArrayImageDataset(archive[f"{split}_images"], labels, transform_of[split], names)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            generator=_loader_generator(seed) if shuffle else None,
            worker_init_fn=_worker_init if num_workers else None,
            drop_last=_drop_last(len(dataset), batch_size, shuffle), pin_memory=False,
            persistent_workers=False)
    return loaders


def build_model(architecture: str, num_classes: int, weights_path: str | Path) -> nn.Module:
    """Build MobileNetV3-Small or EfficientNet-B0 with local pretrained weights.

    Load the bundled ImageNet state dictionary before replacing the final classifier
    for ``num_classes``. Do not request network downloads.
    """
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"architecture must be one of {SUPPORTED_ARCHITECTURES}, got {architecture!r}")
    model = getattr(torchvision.models, architecture)(weights=None)
    state_dict = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    head_index = len(model.classifier) - 1
    old_head = model.classifier[head_index]
    model.classifier[head_index] = nn.Linear(old_head.in_features, int(num_classes))
    model.a3_head_parameters = tuple(
        f"classifier.{head_index}.{name}" for name, _ in model.classifier[head_index].named_parameters())
    model.a3_architecture = architecture
    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze every parameter except the replacement classification head."""
    head = set(getattr(model, "a3_head_parameters", ()) or ())
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (name in head) if head else ("classifier" in name)
    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Make every parameter trainable again (the full fine-tuning condition)."""
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device) -> dict[str, float]:
    """Train for one complete epoch and return sample-weighted ``loss`` and ``accuracy``."""
    model.to(device).train()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        count = labels.numel()
        total_loss += float(loss.detach()) * count
        total_correct += int((logits.detach().argmax(dim=1) == labels).sum())
        total_seen += count
    if total_seen == 0:
        return {"loss": float("nan"), "accuracy": float("nan")}
    return {"loss": total_loss / total_seen, "accuracy": total_correct / total_seen}


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> dict[str, Any]:
    """Evaluate without gradients; return loss, accuracy, labels, predictions, probabilities."""
    model.to(device).eval()
    total_loss, total_seen = 0.0, 0
    all_labels, all_predictions, all_probabilities = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        count = labels.numel()
        total_loss += float(loss.detach()) * count
        total_seen += count
        probabilities = torch.softmax(logits, dim=1)
        all_probabilities.append(probabilities.cpu().numpy())
        all_predictions.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    labels_array = np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.int64)
    predictions_array = np.concatenate(all_predictions) if all_predictions else np.empty(0, dtype=np.int64)
    probabilities_array = (np.concatenate(all_probabilities) if all_probabilities
                           else np.empty((0, 0), dtype=np.float32))
    accuracy = float((predictions_array == labels_array).mean()) if total_seen else float("nan")
    return {"loss": total_loss / total_seen if total_seen else float("nan"),
            "accuracy": accuracy, "labels": labels_array,
            "predictions": predictions_array, "probabilities": probabilities_array}


@torch.no_grad()
def predict(model: nn.Module, images: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    """Return CPU ``predictions`` and softmax ``probabilities`` for one image batch."""
    model.to(device).eval()
    logits = model(images.to(device))
    probabilities = torch.softmax(logits, dim=1)
    return {"predictions": logits.argmax(dim=1).cpu(), "probabilities": probabilities.cpu()}


@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return penultimate-layer embeddings and labels in loader order as NumPy arrays.

    The penultimate-layer embedding is the input to the final classifier layer,
    ``model.classifier[-1]``.
    """
    model.to(device).eval()
    captured: list[np.ndarray] = []

    def hook(_module, inputs, _output):
        captured.append(inputs[0].detach().flatten(start_dim=1).cpu().numpy())

    handle = model.classifier[-1].register_forward_hook(hook)
    all_labels = []
    try:
        for images, labels in loader:
            model(images.to(device))
            all_labels.append(np.asarray(labels).reshape(-1))
    finally:
        handle.remove()
    embeddings = np.concatenate(captured) if captured else np.empty((0, 0), dtype=np.float32)
    labels_array = np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.int64)
    return embeddings, labels_array.astype(np.int64)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray,
                           class_names: list[str]) -> dict[str, Any]:
    """Return confusion matrix, overall accuracy, and per-class precision, recall, F1, and accuracy.

    Return a dictionary with keys ``confusion_matrix`` (rows are true classes, columns
    are predicted classes), ``accuracy``, and per-class arrays ``precision``, ``recall``,
    ``f1``, and ``per_class_accuracy`` in ``class_names`` order.

    Per-class accuracy is ``TP_i / (TP_i + FN_i)``: correct predictions among
    samples whose true label is ``i``. For single-label classification, this is
    numerically identical to per-class recall; report both values.
    """
    labels = np.asarray(labels).astype(int).reshape(-1)
    predictions = np.asarray(predictions).astype(int).reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    count = len(class_names)
    matrix = np.zeros((count, count), dtype=np.int64)
    for true_label, predicted_label in zip(labels, predictions):
        matrix[true_label, predicted_label] += 1

    true_positive = np.diag(matrix).astype(np.float64)
    predicted_total = matrix.sum(axis=0).astype(np.float64)
    actual_total = matrix.sum(axis=1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted_total > 0, true_positive / predicted_total, 0.0)
        recall = np.where(actual_total > 0, true_positive / actual_total, 0.0)
        denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * precision * recall / denominator, 0.0)
    accuracy = float(true_positive.sum() / matrix.sum()) if matrix.sum() else float("nan")
    return {"confusion_matrix": matrix, "accuracy": accuracy,
            "precision": precision, "recall": recall, "f1": f1,
            "per_class_accuracy": recall.copy(), "support": actual_total.astype(np.int64),
            "class_names": list(class_names),
            "macro_f1": float(f1.mean()) if count else float("nan")}


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str],
                          *, normalize: bool = False) -> tuple[plt.Figure, plt.Axes]:
    """Create a labeled confusion-matrix figure without displaying or saving it.

    Rows are true classes and columns are predicted classes. Return ``(fig, ax)``.
    """
    matrix = np.asarray(matrix)
    shown = matrix.astype(np.float64)
    if normalize:
        row_totals = shown.sum(axis=1, keepdims=True)
        shown = np.divide(shown, row_totals, out=np.zeros_like(shown), where=row_totals > 0)

    side = max(5.0, 0.62 * len(class_names) + 2.6)
    fig, ax = plt.subplots(figsize=(side, side * 0.86), dpi=130)
    image = ax.imshow(shown, cmap="Blues", vmin=0.0, vmax=shown.max() if shown.size and shown.max() > 0 else 1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                 label="fraction of true class" if normalize else "samples")

    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(("Normalized confusion matrix" if normalize else "Confusion matrix")
                 + "\n(rows = true, columns = predicted)")

    threshold = (shown.max() if shown.size else 0.0) * 0.6
    if len(class_names) <= 12:
        for row in range(shown.shape[0]):
            for column in range(shown.shape[1]):
                value = shown[row, column]
                ax.text(column, row, f"{value:.2f}" if normalize else f"{int(matrix[row, column])}",
                        ha="center", va="center", fontsize=8,
                        color="white" if value > threshold else "black")
    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    return fig, ax


class HFImageDataset(Dataset):
    """One Hugging Face split wrapped as an image-classification ``Dataset``."""

    def __init__(self, split, image_column: str, label_column: str,
                 class_names: list[str], transform, label_to_index: Mapping | None = None):
        self.split = split
        self.image_column = image_column
        self.label_column = label_column
        self.class_names = list(class_names)
        self.transform = transform
        self.label_to_index = dict(label_to_index) if label_to_index else None

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int):
        row = self.split[int(index)]
        image = row[self.image_column]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        raw_label = row[self.label_column]
        label = self.label_to_index[raw_label] if self.label_to_index is not None else int(raw_label)
        return self.transform(image), int(label)


def _hf_image_column(split) -> str:
    import datasets as hf_datasets
    for name, feature in split.features.items():
        if isinstance(feature, hf_datasets.Image):
            return name
    for name in ("image", "img", "pixel_values"):
        if name in split.features:
            return name
    raise ValueError(f"no image column found among {list(split.features)}")


def build_external_dataloaders(hf_dataset_name: str, label_column: str, batch_size: int,
                               augmentation: str, seed: int, num_workers: int = 0) -> dict[str, DataLoader]:
    """Build deterministic train/validation/test loaders from a Hugging Face Hub dataset.

    Download ``hf_dataset_name`` once via the ``datasets`` library (``datasets.load_dataset``)
    and wrap it with the same augmentation/normalization contract as ``build_dataloaders``:
    ``augmentation`` is ``none``, ``moderate``, or ``strong`` and applies only to training;
    evaluation transforms are deterministic and use ImageNet normalization. Use the dataset's
    own train/validation/test splits when present; for ``marmal88/skin_cancer``, follow
    ``raw = datasets.load_dataset(hf_dataset_name)`` with ``raw = a3_tools.resplit_by_lesion(raw)``,
    which returns lesion-level ``train``/``validation``/``test`` splits. If a split is missing, carve it
    deterministically from ``train`` using ``seed``. ``label_column``
    is the dataset's class-label field (a string category name or an integer index) and must be
    mapped to the same 0..num_classes-1 integer labels the rest of the pipeline expects. Because
    only loaders are returned, keep the ordered class names reachable (for example, as an
    attribute of your dataset) so figures can label each class. Train shuffling is controlled by
    ``seed``. This is the only function permitted to reach the network.
    """
    if augmentation not in AUGMENTATION_LEVELS:
        raise ValueError(f"augmentation must be one of {AUGMENTATION_LEVELS}, got {augmentation!r}")
    import datasets

    raw = datasets.load_dataset(hf_dataset_name)
    if hf_dataset_name == "marmal88/skin_cancer":
        raw = a3_tools.resplit_by_lesion(raw)

    # Carve any missing split deterministically out of train.
    if "validation" not in raw or "test" not in raw:
        pieces = {"train": raw["train"]}
        if "test" in raw:
            pieces["test"] = raw["test"]
        if "validation" in raw:
            pieces["validation"] = raw["validation"]
        needed = [name for name in ("validation", "test") if name not in pieces]
        fraction = 0.15 * len(needed)
        parts = pieces["train"].train_test_split(test_size=fraction, seed=int(seed), shuffle=True)
        pieces["train"] = parts["train"]
        if len(needed) == 2:
            halves = parts["test"].train_test_split(test_size=0.5, seed=int(seed), shuffle=True)
            pieces["validation"], pieces["test"] = halves["train"], halves["test"]
        else:
            pieces[needed[0]] = parts["test"]
        raw = datasets.DatasetDict(pieces)

    feature = raw["train"].features[label_column]
    if isinstance(feature, datasets.ClassLabel):
        class_names, label_to_index = list(feature.names), None
    else:
        observed = set()
        for split in raw.values():
            observed.update(split.unique(label_column))
        class_names = [str(value) for value in sorted(observed)]
        label_to_index = {value: index for index, value in enumerate(sorted(observed))}

    image_column = _hf_image_column(raw["train"])
    transform_of = {"train": training_transform(augmentation),
                    "validation": evaluation_transform(), "test": evaluation_transform()}
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        dataset = HFImageDataset(raw[split], image_column, label_column, class_names,
                                 transform_of[split], label_to_index)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            generator=_loader_generator(seed) if shuffle else None,
            worker_init_fn=_worker_init if num_workers else None,
            drop_last=_drop_last(len(dataset), batch_size, shuffle), pin_memory=False,
            persistent_workers=False)
    return loaders
