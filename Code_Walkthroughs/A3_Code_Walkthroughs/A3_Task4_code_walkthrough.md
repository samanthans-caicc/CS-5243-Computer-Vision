# A3 Task 4 — Code Walkthrough

_Study note (not part of the graded submission). Explains `predict` and
`classification_metrics` in `src/student_code.py`, and the Task 4 notebook cell
(`683c1b07`)._

---

## Key background: one confusion matrix, every metric

Every classification metric in this assignment is a different way of reading **one** table.
The confusion matrix `C` has a row per *true* class and a column per *predicted* class;
`C[i, j]` counts samples of true class `i` that the model called `j`.

For one class `i`:

| Quantity | Read from `C` | Meaning |
|---|---|---|
| `TP_i` | `C[i, i]` (diagonal) | class `i`, called `i` |
| `FN_i` | row `i` total − `TP_i` | class `i`, called something else |
| `FP_i` | column `i` total − `TP_i` | something else, called `i` |

- **Precision** `TP / (TP + FP)` = diagonal / **column** total: *when the model says `i`, how
  often is it right?*
- **Recall** `TP / (TP + FN)` = diagonal / **row** total: *of the real `i` samples, how many
  did it find?*
- **F1** `2PR / (P + R)`: the harmonic mean, which stays low if *either* is low.
- **Per-class accuracy** as the task defines it, `TP_i / (TP_i + FN_i)`, is the same formula
  as recall. For single-label classification they are identical numbers; the task asks for
  both columns so the table matches the wording of the requirement.
- **Overall accuracy** = trace / total.
- **Macro-F1** = the unweighted mean of the per-class F1s. Every class counts equally no
  matter how many samples it has — which is why it collapses on imbalanced data where
  accuracy does not (Experiment 8).

---

## `predict`

```python
@torch.no_grad()
def predict(model, images, device):
    model.to(device).eval()
    logits = model(images.to(device))
    probabilities = torch.softmax(logits, dim=1)
    return {"predictions": logits.argmax(dim=1).cpu(), "probabilities": probabilities.cpu()}
```

- The single-batch counterpart of `evaluate_model`: same `no_grad` + `eval()` pair, no
  labels, no loss.
- `softmax(dim=1)` normalizes across the **class** axis (dim 0 is the batch).
- Predictions come from the logits' argmax. Softmax preserves order, so this always equals the
  probabilities' argmax — the notebook checks it (`argmax agrees: True`).
- Results come back on the **CPU** as tensors (the task asks for CPU outputs), so callers can
  convert to NumPy or print them without device errors.

---

## `classification_metrics`

```python
labels = np.asarray(labels).astype(int).reshape(-1)
predictions = np.asarray(predictions).astype(int).reshape(-1)
if labels.shape != predictions.shape: raise ValueError(...)
count = len(class_names)
matrix = np.zeros((count, count), dtype=np.int64)
for true_label, predicted_label in zip(labels, predictions):
    matrix[true_label, predicted_label] += 1
```

- `reshape(-1)` accepts lists, column vectors, or tensors moved to NumPy.
- The matrix size comes from `class_names`, **not** from the labels. If a class never appears
  in a small test set (or is never predicted), its row or column still exists — essential for
  the dermatofibroma row in Experiment 8, which has support but zero correct predictions.
- The Python loop is O(N) and fine for N ≤ 5,400. `np.add.at(matrix, (labels, predictions), 1)`
  is the vectorized equivalent.

```python
true_positive   = np.diag(matrix).astype(np.float64)
predicted_total = matrix.sum(axis=0).astype(np.float64)   # column sums
actual_total    = matrix.sum(axis=1).astype(np.float64)   # row sums
with np.errstate(divide="ignore", invalid="ignore"):
    precision = np.where(predicted_total > 0, true_positive / predicted_total, 0.0)
    recall    = np.where(actual_total > 0,    true_positive / actual_total,    0.0)
    denominator = precision + recall
    f1 = np.where(denominator > 0, 2 * precision * recall / denominator, 0.0)
```

- **`axis=0` sums down columns (predicted totals); `axis=1` sums across rows (true totals).**
  Swapping these swaps precision and recall — no error, just wrong numbers. With rows = true,
  columns = predicted, the axis names follow directly.
- **Zero-division policy: 0.** A class that is never predicted has no precision defined;
  reporting 0 (scikit-learn's `zero_division=0` behaviour) is conservative and keeps the arrays
  free of `NaN`, so macro-F1 still computes. `np.where` evaluates both branches, so the
  divide-by-zero still happens internally; `np.errstate` silences the warning and `np.where`
  discards the result.
- Computed in `float64` so per-class values of large splits do not lose precision.

```python
return {"confusion_matrix": matrix, "accuracy": ..., "precision": precision, "recall": recall,
        "f1": f1, "per_class_accuracy": recall.copy(), "support": actual_total.astype(np.int64),
        "class_names": list(class_names), "macro_f1": float(f1.mean())}
```

- The required keys plus three extras the notebook uses: `support` (samples per true class),
  `class_names`, and `macro_f1`.
- `per_class_accuracy` is a **copy** of `recall`, so modifying one array can never silently
  change the other.

---

## The notebook cell logic (`683c1b07`)

On the one-epoch MobileNet from Task 3:

```
overall accuracy 0.8278 | confusion matrix (6, 6) | row sums (true-class support) [30, 30, 30, 30, 30, 30]
     class  support  precision  recall     f1  per_class_accuracy
   checker       30     0.9375  1.0000 0.9677              1.0000
   circles       30     0.9677  1.0000 0.9836              1.0000
  diagonal       30     0.6500  0.4333 0.5200              0.4333
      dots       30     0.7143  1.0000 0.8333              1.0000
horizontal       30     0.8421  0.5333 0.6531              0.5333
  vertical       30     0.8333  1.0000 0.9091              1.0000
per_class_accuracy equals recall for this single-label task: True
```

Reading it back to counts shows how precision and recall describe different errors:

- **diagonal**: recall 0.4333 → 13 of 30 found. Precision 0.65 → the model said "diagonal"
  20 times, 13 correctly. It misses most diagonals *and* sometimes calls other things diagonal.
- **dots**: recall 1.0 but precision 0.7143 → it said "dots" 42 times for 30 real dots. Twelve
  other images were dumped into "dots". Perfect recall, poor precision.
- The predicted totals (32 + 31 + 20 + 42 + 19 + 36) sum to 180, as they must.

An undertrained model is the useful case here: with Experiment 6's 100 % model every entry is
1.0 and the table cannot show the difference between the metrics.

---

## Two concepts examiners love to probe

1. **Why can a class have recall 1.0 and still be a problem?** Recall only looks at that class's
   row. "dots" above finds every real dot, but its column holds 12 intruders. Precision is the
   column view and catches it; F1 (0.83) combines both.
2. **Why does macro-F1 fall so much further than accuracy on imbalanced data?** Accuracy is
   weighted by support, so a dominant class carries it. Macro-F1 weights each class equally, so
   one class with F1 = 0 costs 1/K of the score no matter how few samples it has. In
   Experiment 8 that is the difference between 0.795 accuracy and 0.479 macro-F1.
