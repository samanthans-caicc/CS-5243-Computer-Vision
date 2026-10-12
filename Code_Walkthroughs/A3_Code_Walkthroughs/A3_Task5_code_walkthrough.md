# A3 Task 5 — Code Walkthrough

_Study note (not part of the graded submission). Explains `extract_embeddings` and
`plot_confusion_matrix` in `src/student_code.py`, and the Task 5 notebook cell
(`36e96350`)._

---

## Key background: what a "penultimate-layer embedding" is

The final layer of a classifier is a linear map: `logits = W · z + b`. Its input `z` is the
**penultimate-layer embedding** — the feature vector the whole network was built to produce.
Everything the head knows about an image is in `z`; the head only draws linear boundaries
through that space.

So looking at `z` directly shows *what the network learned*, separately from the decision it
makes. If classes form tight, separate clusters in `z`, a linear head can split them easily.
If two classes overlap in `z`, no head trained on top of it can tell them apart — that is
where confusions come from.

| Architecture | Final layer | Embedding `z` |
|---|---|---|
| MobileNetV3-Small | `classifier[3] = Linear(1024, 6)` | 1024-d output of `Linear(576,1024) → Hardswish → Dropout` |
| EfficientNet-B0 | `classifier[1] = Linear(1280, 6)` | 1280-d pooled backbone features after `Dropout` |

In eval mode dropout is the identity, so `z` is deterministic.

---

## `extract_embeddings`

```python
@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.to(device).eval()
    captured = []

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
```

**Why a forward hook?** The embedding is an intermediate value that the model's `forward`
never returns. A forward hook is a callback PyTorch runs every time a module finishes its
forward pass, receiving `(module, inputs, output)`. Hooking the **final** layer and keeping
its **input** captures `z` exactly as the head sees it, for both architectures, without
re-implementing either network's forward pass or slicing `classifier` by hand.

- `inputs` is a tuple (a module can take several arguments); `inputs[0]` is the tensor.
- `.flatten(start_dim=1)` keeps the batch axis and flattens the rest, so the result is always
  `(batch, features)` even if a model feeds the head a 4-D tensor.
- `.detach().cpu().numpy()` copies out of the graph and off the GPU immediately, so captured
  batches do not accumulate GPU memory.
- **`try / finally: handle.remove()`** — a hook stays attached to the module until removed.
  If it were left behind, every later forward pass (training, evaluation, the next call to
  this function) would keep appending to a list nobody reads. `finally` removes it even if a
  batch raises.
- Labels come from the **loader**, not the model, in the same batch order as the captures.
  Since validation and test loaders never shuffle, the returned labels equal the ones
  `evaluate_model` returns — the notebook checks this.
- The public test checks the contract with a tiny `Linear(12,5) → ReLU → Linear(5,3)` model:
  the embedding must be the 5-d **post-ReLU** activation, which is the input to the last layer.

---

## `plot_confusion_matrix`

```python
matrix = np.asarray(matrix)
shown = matrix.astype(np.float64)
if normalize:
    row_totals = shown.sum(axis=1, keepdims=True)
    shown = np.divide(shown, row_totals, out=np.zeros_like(shown), where=row_totals > 0)

side = max(5.0, 0.62 * len(class_names) + 2.6)
fig, ax = plt.subplots(figsize=(side, side * 0.86), dpi=130)
image = ax.imshow(shown, cmap="Blues", vmin=0.0, vmax=shown.max() if ... else 1.0)
fig.colorbar(image, ax=ax, ..., label="fraction of true class" if normalize else "samples")
ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
ax.set_yticks(range(len(class_names)), class_names)
ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
```

- **Rows = true, columns = predicted** follows from `imshow`: array row 0 is drawn at the top,
  and the axis labels state the convention in the figure itself — the title repeats it.
- **Normalization is by row** (`axis=1`): each cell becomes "fraction of true class `i`
  predicted as `j`", so the diagonal is per-class recall. `where=row_totals > 0` with a zero
  `out` array keeps an empty class at 0 instead of `NaN`.
- `keepdims=True` keeps the totals as a column `(K, 1)` so the division broadcasts across each row.
- Figure size grows with the class count (`0.62` inch per class) so 6, 7, and 10 classes all
  stay legible; `dpi=130` puts the saved figure well above the validator's 500×400 minimum.

```python
threshold = (shown.max() if shown.size else 0.0) * 0.6
if len(class_names) <= 12:
    for row ...: for column ...:
        ax.text(column, row, f"{value:.2f}" if normalize else f"{int(matrix[row, column])}",
                ..., color="white" if value > threshold else "black")
ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.6)
fig.tight_layout()
return fig, ax
```

- Every cell is annotated with its count (or fraction). Text switches to white above 60 % of
  the maximum so it stays readable on dark blue cells. Above 12 classes the numbers would
  overlap, so annotation is skipped.
- The **minor ticks** at half-integer positions draw white gridlines *between* cells, not
  through their centres.
- **No `plt.show()`, no `savefig`.** The function only builds and returns `(fig, ax)`. The
  caller decides whether to save it, add to it, or close it — Experiment 6 saves it as
  `confusion_matrix.png`.

---

## The notebook cell logic (`36e96350`)

```
extract_embeddings -> (180, 1024) (180,) | penultimate layer feeds Linear(in_features=1024)
labels returned in loader order: True
plot_confusion_matrix -> ('Figure', 'Axes') | nothing written to disk by the function itself
```

- `(180, 1024)`: one 1024-d MobileNet embedding per validation image, and the width matches the
  head's `in_features`.
- The label check compares against `demo_val["labels"]` from Task 3's `evaluate_model` — two
  different passes over the same loader agree on order.
- The cell ends with `plt.close(demo_figure)` followed by the bare expression `demo_figure`.
  Jupyter renders the returned figure as the cell's result. Without the `close`, the inline
  backend would *also* render every still-open pyplot figure at the end of the cell, and the
  figure would appear twice — which is what the first version of the notebook did in this
  cell and in every Section 5 cell that ends with `figure`.

---

## Two concepts examiners love to probe

1. **Why hook the input to the last layer rather than the output of the second-to-last
   module?** For EfficientNet the module before `classifier[1]` is a `Dropout` and for
   MobileNet it is also `Dropout` — reading that module's output works too, but only by
   knowing each architecture's layout. "Input to `classifier[-1]`" is one definition that is
   correct for both, and for any future model with a `classifier` sequence.
2. **What does a row-normalized confusion matrix show that a raw one hides?** With unequal
   class sizes, raw counts make big classes look dark everywhere. Normalizing each row to sum
   to 1 puts every class on the same scale, so the diagonal reads directly as per-class recall
   — the view Experiments 8 and 9 rely on.
