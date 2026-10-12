# A3 Experiments 5–6 — Code Walkthrough

_Study note (not part of the graded submission). Explains the Experiments 5–6 notebook cell
(`0b8da407`): training the validation-selected final model, the single held-out test
evaluation, `confusion_matrix.png`, `embedding_visualization.png`, and `predictions.csv`. It
uses `classification_metrics`, `plot_confusion_matrix`, and `extract_embeddings` (Tasks 4–5)
plus the harness's `run_training`, `test_report`, and `per_class_table`._

---

## Key background: select on validation, then look at test exactly once

Experiments 1–4 each fixed one factor using validation evidence only. This cell combines the
four winners into one configuration, trains it, and **then** evaluates on the test split —
once. The test result is reported, never acted on: nothing after this cell changes the
configuration because of a test number. That is what makes it an honest estimate of how the
selected model performs on unseen data.

Two separate questions are answered here:

- **Experiment 5 — what did the model learn?** The penultimate embeddings of the test images,
  projected to 2-D with PCA.
- **Experiment 6 — how well does it classify?** Accuracy, per-class precision / recall / F1 /
  per-class accuracy, and the confusion matrix, on test.

---

## The notebook cell logic (`0b8da407`)

### The final configuration

```python
SELECTED_CONFIG = {"architecture": BEST_ARCHITECTURE, "mode": BEST_MODE,
                   "augmentation": BEST_AUGMENTATION, "optimizer_name": BEST_OPTIMIZER}
SELECTED_BY = ("factor-wise winners of Experiments 1-4, each chosen on validation accuracy with "
               "validation loss as tie-break; the reported epoch is the best validation epoch. "
               "The test split was not consulted for any selection decision.")
FINAL_RUN = run_training("final", epochs=EPOCHS, seed=SEED, **SELECTED_CONFIG)
```

- `SELECTED_CONFIG` is built from the four globals, not typed by hand, so it cannot drift from
  what the experiments chose: `efficientnet_b0`, `frozen`, `none`, `sgd`.
- `SELECTED_BY` is the sentence written into `model_metadata.json` (Experiment 7). The README
  requires the selection description to name **validation** evidence; this states the rule,
  the tie-break, and the epoch choice.
- `**SELECTED_CONFIG` unpacks the dictionary as keyword arguments, and Task 7 reuses the same
  dictionary for its baseline.
- **Factor-wise selection** means each factor was chosen with the earlier winners fixed. It is
  cheaper than a full grid (2 × 2 × 3 × 2 = 24 runs) but cannot detect interactions — for
  example, a factor that only wins in combination with a different level of another.
- This configuration is identical to `exp4_sgd`, and with the same seed it reproduces it
  exactly: best validation 1.000 at epoch 2, validation loss 0.0382.

### One test evaluation

```python
test_evaluation, test_metrics = test_report(FINAL_RUN)
record(6, "held-out test evaluation", "final", "test_accuracy", test_metrics["accuracy"], "fraction")
...
for index, class_name in enumerate(CLASS_NAMES):
    for metric in ("precision", "recall", "f1", "per_class_accuracy"):
        record(6, ..., class_name, f"test_{metric}", test_metrics[metric][index], "fraction")
```

- `test_report` evaluates the **restored best-validation checkpoint** (epoch 2), not the epoch-10
  weights.
- Per-class rows use the **class name** as the `condition`, so the CSV can be filtered by
  class: 3 overall rows + 6 classes × 4 metrics = 27 Experiment 6 records.

### `confusion_matrix.png`

`sc.plot_confusion_matrix(test_metrics["confusion_matrix"], CLASS_NAMES)` — the graded
function, unnormalized, saved, then closed at the end of the cell so only the embedding figure
is displayed.

### `embedding_visualization.png`

```python
embeddings, embedding_labels = sc.extract_embeddings(FINAL_RUN["model"], FINAL_RUN["loaders"]["test"], DEVICE)
projection = PCA(n_components=2, random_state=SEED)
points = projection.fit_transform(embeddings)
variance = projection.explained_variance_ratio_
```

- `extract_embeddings` returns a `(180, 1280)` array — EfficientNet's pooled features feeding
  the head.
- **PCA** finds the two orthogonal directions of largest variance in the 1280-d cloud and
  projects every point onto them. It is linear and deterministic (the `random_state` only
  matters for randomized solvers), so the plot is reproducible.
- Each axis label carries its **explained-variance ratio**, so the figure states how much of the
  embedding it is actually showing.
- Points are coloured by **true** class, so the figure shows how the representation organizes
  the classes, independent of the head's decisions.
- PCA is **unsupervised**: it does not see the labels, and it is fitted on test embeddings only
  for visualization — it has no role in training or selection.

### `predictions.csv`

```python
predictions_frame = pd.DataFrame({
    "sample_id": [f"test_{index:04d}" for index in range(len(test_evaluation["labels"]))],
    "split": "test",
    "true_label": [CLASS_NAMES[index] for index in test_evaluation["labels"]],
    "predicted_label": [CLASS_NAMES[index] for index in test_evaluation["predictions"]],
    "confidence": test_evaluation["probabilities"].max(axis=1).round(6)})
```

- One row per test sample, in the README column order. Because the test loader never shuffles,
  `test_0000` is the first image in the archive's test array, and so on.
- Labels are written as **names**, not integers, so the file is readable on its own.
- `confidence` is the largest softmax probability — the probability of the predicted class.

---

## What it produced

```
test accuracy 1.0000 | macro F1 1.0000 | test loss 0.0322
(every class: support 30, precision = recall = f1 = per_class_accuracy = 1.0)
```

- Every one of the 180 test images is classified correctly; the confusion matrix is diagonal.
- Validation saturated at 1.000 from Experiment 1 onward, and test does too. The per-class table
  is complete as required, but on this dataset it cannot distinguish one metric from another —
  Task 4's demo and Experiment 8 are where those differences are visible.
- Confidence is not 1.0 everywhere: the first three rows have 0.70, 0.84, and 0.94. The mean
  test confidence is 0.970 (reported in Experiment 7 at σ = 0).
- **The embedding plot** (PC 1 = 6.6 %, PC 2 = 5.6 %, together **12.2 %** of the variance):
  *circles* and *dots* form two clearly separate clusters; *checker*, *diagonal*, *horizontal*,
  and *vertical* overlap in one region, with horizontal and vertical the most intermixed. The
  head separates all six perfectly, so the overlap is in the 2-D projection, not necessarily in
  the full 1280-d space — 88 % of the variance is in directions the plot does not show.

---

## Two concepts examiners love to probe

1. **Why does a 2-D PCA show overlap for classes the model never confuses?** PCA keeps the
   directions of *largest variance*, which need not be the directions that *separate classes*.
   A linear head can use any of the 1280 dimensions; a class boundary can sit entirely in
   directions PCA ranked low. A supervised projection (e.g. LDA) or t-SNE would show separation
   PCA hides — at the cost of being harder to interpret.
2. **Why is checking the test set "once" a rule and not a suggestion?** Each look at test that
   influences a decision turns test into a second validation set and biases the estimate
   upward. The harness enforces it in code: `run_training` never reads `loaders["test"]`, and
   every `test_report` call follows a completed selection.
