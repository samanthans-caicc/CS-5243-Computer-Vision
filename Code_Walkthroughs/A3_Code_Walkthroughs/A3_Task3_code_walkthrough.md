# A3 Task 3 — Code Walkthrough

_Study note (not part of the graded submission). Explains `freeze_backbone`, `unfreeze_all`,
`train_one_epoch`, and `evaluate_model` in `src/student_code.py`, and the Task 3 notebook
cell (`bcda558b`)._

---

## Key background: the two ways to adapt a pretrained network

- **Frozen transfer (a "linear probe").** Keep every pretrained weight fixed; train only the
  new head. The backbone acts as a fixed feature extractor and the head learns a linear
  boundary in its feature space. Cheap — 6,150 or 7,686 trainable numbers — and hard to
  overfit, but it can only use features the backbone already computes.
- **Full fine-tuning.** Every weight is trainable. The backbone can reshape its features for
  the new task, which helps when the new images look unlike ImageNet (Experiment 9). It costs
  more per step and, with millions of free weights and a small dataset, can memorize
  (Task 6, Failure A).

In PyTorch, "frozen" means `requires_grad = False`: autograd computes no gradient for that
tensor, and the optimizer — which is only handed tensors with `requires_grad = True` — never
updates it.

A training loop has two modes that must not be confused. **Training mode** (`model.train()`)
turns on dropout and makes BatchNorm normalize with the *current batch's* statistics while
updating its running averages. **Evaluation mode** (`model.eval()`) turns dropout off and makes
BatchNorm use the stored running averages. Evaluation must use eval mode **and** disable
gradient tracking.

---

## `freeze_backbone` and `unfreeze_all`

```python
def freeze_backbone(model):
    head = set(getattr(model, "a3_head_parameters", ()) or ())
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (name in head) if head else ("classifier" in name)
    return model

def unfreeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model
```

- **The primary rule is exact names.** `build_model` stored the head's parameter names, so
  only those tensors stay trainable. For MobileNet this matters: `classifier.0`
  (`Linear(576, 1024)`) is pretrained and must be frozen even though it lives in `classifier`.
- **The fallback** (`"classifier" in name`) only applies to a model that did not come from
  `build_model` — for example the public test's tiny module — and satisfies the public test's
  check that every trainable name contains `"classifier"`.
- Both functions **set** the flag on every parameter instead of only flipping some. Calling
  `freeze_backbone` on a half-frozen model gives the same result as calling it on a fresh one.
- `unfreeze_all` exists so the notebook's `run_training` can express both Experiment 2
  conditions as one line: `freeze_backbone(model) if mode == "frozen" else unfreeze_all(model)`.
- Both return the model, so calls can be chained, but they also modify it in place.

---

## `train_one_epoch`

```python
model.to(device).train()
total_loss, total_correct, total_seen = 0.0, 0, 0
for images, labels in loader:
    images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    count = labels.numel()
    total_loss += float(loss.detach()) * count
    total_correct += int((logits.detach().argmax(dim=1) == labels).sum())
    total_seen += count
return {"loss": total_loss / total_seen, "accuracy": total_correct / total_seen}
```

- The five-step core: **zero → forward → loss → backward → step.** Forgetting `zero_grad`
  would add each batch's gradient to the previous one's. `set_to_none=True` frees the gradient
  tensors instead of filling them with zeros — slightly faster, and a frozen parameter's
  `.grad` stays `None`.
- **Sample-weighted averaging.** `CrossEntropyLoss` returns the *mean* over the batch, so the
  code multiplies it back by the batch size before summing. If the last batch is smaller, a
  plain average of batch means would over-weight it; this gives the true per-sample mean.
- `.detach()` and `float(...)` take the value out of the autograd graph so the running total
  does not keep every batch's graph alive in memory.
- Accuracy is `argmax` of the logits — softmax is monotonic, so it is unnecessary for the
  winning class.
- The empty-loader guard returns `NaN` instead of dividing by zero.

**Training accuracy is a running average, not a clean measurement.** It is computed batch by
batch while the weights are still changing, in training mode (dropout on, augmentation
applied). That is why the notebook often shows *negative* train–validation gaps (training
accuracy below validation): a clean evaluation of the end-of-epoch model can beat the running
average of the model as it improved during the epoch.

**Frozen does not mean nothing changes.** `model.train()` applies to the whole model,
including the frozen backbone. Its BatchNorm layers still update their running statistics to
the new dataset, and the backbone's dropout is active. Only the learnable weights are fixed.

---

## `evaluate_model`

```python
@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.to(device).eval()
    for images, labels in loader:
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.detach()) * count
        probabilities = torch.softmax(logits, dim=1)
        all_probabilities.append(probabilities.cpu().numpy())
        all_predictions.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    ...
    return {"loss": ..., "accuracy": ..., "labels": ..., "predictions": ..., "probabilities": ...}
```

- **Two separate switches, both needed.** `@torch.no_grad()` disables gradient tracking for
  the whole call (memory and speed). `.eval()` changes layer *behaviour* (dropout off,
  BatchNorm uses running statistics). One does not imply the other.
- The same sample-weighted loss as training, but accuracy is computed from the concatenated
  arrays — the same thing, written the direct way.
- The returned arrays are NumPy in **loader order**. Everything downstream (metrics,
  `predictions.csv`, embeddings, robustness) relies on that order matching the dataset.
- The loss uses whatever criterion is passed in. The notebook always passes **plain**
  cross-entropy here, even when training used label smoothing (Task 7), so validation losses
  are comparable across conditions.

---

## The notebook cell logic (`bcda558b`)

The cell trains one MobileNet epoch with a frozen backbone and checks each contract:

```
trainable parameters 1,524,006 -> 6,150 after freeze_backbone
still trainable: ['classifier.3.weight', 'classifier.3.bias']
train_one_epoch -> {'loss': 0.8152, 'accuracy': 0.8333}
evaluate_model  -> {'loss': 1.0232, 'accuracy': 0.8278, 'labels': (180,), 'predictions': (180,), 'probabilities': (180, 6)}
probabilities rows sum to 1: True | gradients disabled during evaluation: True
```

- Only the replacement head survives the freeze — exactly two tensors, 6,150 numbers.
- The optimizer is built from `[p for p in demo_model.parameters() if p.requires_grad]`, so
  Adam has no state for frozen tensors at all.
- One epoch already reaches ~83 % validation accuracy; the model is reused by Tasks 4 and 5 as
  an imperfect model with real confusions to show.

### How the cell checks that gradients are off

The check has to observe the flag **while `evaluate_model` is running**, because the
`@torch.no_grad()` decorator turns gradients off only for the duration of the call and restores
them afterwards. The first version of this cell called `torch.is_grad_enabled()` after the
function had returned, so it always saw `True` and printed `gradients disabled during
evaluation: False` — even though `evaluate_model` does disable gradients.

The cell now registers a forward hook that records the flag on every forward pass inside the
call, then removes it:

```python
grad_flags = []  # recorded inside evaluate_model's forward passes
probe = demo_model.register_forward_hook(lambda *_: grad_flags.append(torch.is_grad_enabled()))
demo_val = sc.evaluate_model(demo_model, demo["validation"], criterion, DEVICE)
probe.remove()
... bool(grad_flags) and not any(grad_flags)
```

`bool(grad_flags)` guards against a vacuous `True`: if the hook never fired, the check fails
instead of passing. The hook is attached *after* `train_one_epoch`, so it only sees evaluation
passes.

---

## Two concepts examiners love to probe

1. **Why are `model.eval()` and `torch.no_grad()` both needed?** `no_grad` controls autograd
   (no graph, no memory for gradients); `eval` controls layer behaviour (dropout, BatchNorm).
   Evaluating under `no_grad` in training mode gives randomly dropped activations and corrupts
   BatchNorm's running statistics with validation data; `eval` without `no_grad` gives correct
   numbers while wasting memory on a graph that is never used.
2. **Why multiply the batch loss by the batch size?** Because the loss is a batch *mean*.
   Summing `mean × count` and dividing by the total count gives the mean over all samples,
   regardless of how unevenly the last batch is sized.
