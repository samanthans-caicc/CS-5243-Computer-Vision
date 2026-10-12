# A3 Task 2 — Code Walkthrough

_Study note (not part of the graded submission). Explains `build_model` in
`src/student_code.py` and the Task 2 notebook cell (`a3bc3d4b`)._

---

## Key background: what transfer learning reuses, and what it replaces

An ImageNet classifier is two pieces glued together:

- a **backbone** (`model.features` + pooling) that turns a 224×224 image into a feature
  vector — 576 → 1024 numbers for MobileNetV3-Small, 1280 for EfficientNet-B0; and
- a **head** (`model.classifier`) that maps that vector to **1000** ImageNet class scores.

The backbone's filters (edges, textures, repeating structures) are general; the head's
1000 outputs are specific to ImageNet's classes and useless for six pattern classes. Transfer
learning keeps the backbone's trained weights and swaps the **last layer** of the head for a
fresh `Linear(in_features → 6)`.

The order matters: **load the pretrained weights first, then replace the head.** The saved
state dictionary describes the 1000-class model; it only loads cleanly into a model of that
exact shape. Replace the head first and `load_state_dict` fails on a shape mismatch.

The assignment is offline. torchvision normally downloads weights when asked; here the
weights ship with the assignment and the code must never ask torchvision to fetch anything.

---

## `build_model`

```python
def build_model(architecture, num_classes, weights_path):
    if architecture not in SUPPORTED_ARCHITECTURES: raise ValueError(...)
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
```

**Step 1 — build the bare architecture.** `getattr(torchvision.models, "efficientnet_b0")`
is the constructor function `torchvision.models.efficientnet_b0`. Calling it with
`weights=None` builds the network with **random** initialization and no download. The
validated name list doubles as protection against `getattr` reaching an arbitrary attribute.

**Step 2 — load the local ImageNet weights.**

- `map_location="cpu"` — the weights might have been saved from a GPU; loading to CPU works on
  any machine. The model is moved to the device later, in the training functions.
- `weights_only=True` — `torch.load` uses pickle, and pickle can execute code. This flag
  restricts loading to tensors and plain containers. It is the safe default for any file you
  did not produce in the same session.
- `load_state_dict(state_dict)` uses the default `strict=True`: **every** key must match, in
  both directions. A missing BatchNorm buffer or a renamed layer would raise instead of
  silently leaving a layer random. Strictness is what turns "the weights loaded" from a hope
  into a checked fact.

**Step 3 — replace only the last head layer.** The two classifier heads differ:

| Architecture | `model.classifier` | `head_index` | Replacement |
|---|---|---|---|
| MobileNetV3-Small | `Linear(576,1024)`, `Hardswish`, `Dropout`, `Linear(1024,1000)` | 3 | `Linear(1024, 6)` |
| EfficientNet-B0 | `Dropout`, `Linear(1280,1000)` | 1 | `Linear(1280, 6)` |

`len(model.classifier) - 1` finds the last layer for both without an `if architecture == …`
branch, and `old_head.in_features` reads its input width instead of hard-coding 1024/1280.
MobileNet's `Linear(576, 1024)` **keeps its pretrained weights** — only the 1000-way output
layer is new.

The replacement `nn.Linear` gets PyTorch's default (Kaiming-uniform) initialization from the
global `torch` RNG. That is why the notebook's `run_training` calls `seed_everything(seed)`
*before* `build_model`: two conditions with the same seed start from the same random head.

**Step 4 — record the head's parameter names.** `a3_head_parameters` becomes
`("classifier.3.weight", "classifier.3.bias")` for MobileNet. This is consumed by
`freeze_backbone` (Task 3), and it exists for MobileNet specifically: a looser rule like
"anything with `classifier` in its name is trainable" would also unfreeze the pretrained
`classifier.0` layer (590,848 weights), which is not the replacement head.

---

## The notebook cell logic (`a3bc3d4b`)

For each architecture, the cell builds the model, compares its backbone with the weight file,
runs a forward pass on two zero images, and prints the head shape, parameter counts, and the
comparison result.

```
mobilenet_v3_small  logits (2, 6)  Linear(1024 -> 6)  total 1,524,006  trainable 1,524,006  backbone_matches_local_weights True
efficientnet_b0     logits (2, 6)  Linear(1280 -> 6)  total 4,015,234  trainable 4,015,234  backbone_matches_local_weights True
```

- **`logits (2, 6)`** — the new head produces six scores per image.
- **trainable = total** — nothing is frozen yet; freezing is Task 3's job.
- The head sizes check out by hand: `1024·6 + 6 = 6,150` and `1280·6 + 6 = 7,686`, the exact
  trainable counts Task 3 and Experiment 1 report after freezing.
- `downloaded_anything: False` is a hard-coded literal, not a measurement — it states the
  design rather than testing it.

### Why the comparison has to come before the forward pass

A freshly built model is in **training mode**. In training mode every BatchNorm layer updates
its `running_mean`, `running_var`, and `num_batches_tracked` buffers from the batch it sees —
`torch.no_grad()` stops gradients, not buffer updates. The first version of this cell ran the
zero-image forward pass *first*, and printed `backbone_matches_local_weights: False` even though
the load was exact. Checking outside the notebook showed why:

```
right after build:        True
after train-mode forward: False  (102 mismatched keys, all running_mean / running_var / num_batches_tracked)
learnable weights still match: True
```

The cell now compares **before** the forward pass (calling `model.eval()` first would also
work: eval-mode BatchNorm reads its running statistics instead of updating them).

---

## Two concepts examiners love to probe

1. **Why must the state dictionary be loaded before the head is replaced?** The file holds a
   tensor of shape `(1000, 1280)` under `classifier.1.weight`. With `strict=True`, loading it
   into a model whose `classifier.1` is already `Linear(1280, 6)` raises a size-mismatch
   error. Loading first fills the original 1000-class model exactly, then the replacement
   throws away only the ImageNet-specific layer.
2. **What is the difference between a parameter and a buffer?** Parameters (`weight`,
   `bias`) are learned by gradient descent and listed by `model.parameters()`. Buffers
   (BatchNorm's running statistics) are part of `state_dict()` but are updated by forward
   passes in training mode, not by the optimizer. That is why `requires_grad=False` does not
   freeze them — and why the Task 2 check above tripped over them.
