# A3 Task 7 — Code Walkthrough

_Study note (not part of the graded submission). Explains the Task 7 notebook cell
(`84dd2649`): the label-smoothing extension, its controlled baseline, and
`outputs/extension/label_smoothing_comparison.png`. No new graded function — the only change
is the `label_smoothing` argument the harness passes to `nn.CrossEntropyLoss`._

---

## Key background: what label smoothing changes

Ordinary cross-entropy trains toward a **one-hot** target: probability 1 for the true class,
0 for the rest. The loss keeps falling as the true-class probability approaches 1, so the
network is always rewarded for pushing its logits further apart — toward ever more confident
predictions.

**Label smoothing** with parameter ε replaces the target with

```
target = (1 − ε) · one_hot + ε / K
```

For ε = 0.1 and K = 6: **0.9167** for the true class and **0.0167** for each other class. The
loss is now minimized when the predicted distribution *equals* that target, so the model is
no longer rewarded for confidence beyond ~0.92. The intended effects are less over-confidence
(better calibration) and sometimes better generalization; it does not change which class the
target favours.

**Calibration** asks whether confidence matches correctness: among predictions made with 90 %
confidence, are 90 % right? This cell uses the simplest summary of it — mean confidence minus
accuracy, which it calls `overconfidence_gap`. Positive means over-confident; negative means
under-confident.

---

## The notebook cell logic (`84dd2649`)

### A controlled pair

```python
SMOOTHING = 0.1
for label, epsilon in (("baseline_ce", 0.0), (f"label_smoothing_{SMOOTHING:g}", SMOOTHING)):
    EXTENSION_RUNS[label] = run_training(f"ext_{label}", epochs=EPOCHS, seed=SEED,
                                         label_smoothing=epsilon, **SELECTED_CONFIG)
```

- **Both runs use `SELECTED_CONFIG`** — `efficientnet_b0`, frozen, no augmentation, SGD `1e-2` —
  with the same seed and 10 epochs. The *only* argument that differs is `label_smoothing`.
- Inside `run_training`, smoothing applies to the **training** criterion only.
  `eval_criterion` is always plain cross-entropy, so validation and test losses measure both
  models on the same scale, and checkpoint selection uses the same evidence for both.
- The baseline is the same configuration as the final model and reproduces it exactly (test
  loss 0.0322, mean confidence 0.9702), so it is also a third reproduction check.

### The comparison

```python
evaluation, metrics = test_report(run)
confidence = evaluation["probabilities"].max(axis=1)
correct = evaluation["predictions"] == evaluation["labels"]
row = {..., "mean_confidence": float(confidence.mean()),
       "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else float("nan"),
       "mean_confidence_wrong":   float(confidence[~correct].mean()) if (~correct).any() else float("nan"),
       "overconfidence_gap": float(confidence.mean() - metrics["accuracy"])}
```

- Confidence is the max softmax probability per test image. Splitting it into correct and wrong
  predictions is a standard calibration check: a well-calibrated model is less confident when it
  is wrong. The guards return `NaN` when a group is empty instead of averaging nothing.
- Five metrics per condition plus two deltas (smoothed − baseline) are recorded: 12 `task7`
  rows. The `"cross-entropy" if metric == "test_loss"` branch never fires, because `test_loss`
  is not in that loop's list — harmless, but dead code.

### `label_smoothing_comparison.png`

Three panels, saved to `outputs/extension/` as the single declared extension artifact:
validation accuracy per epoch (the selection evidence, which must look the same for the
comparison to be fair), test accuracy beside mean confidence for each condition, and
step histograms of the per-image confidence distributions.

---

## What it produced

```
          condition  best_val_acc  test_acc  test_loss  mean_conf  mean_conf_wrong  overconfidence_gap
        baseline_ce           1.0       1.0     0.0322     0.9702              NaN             -0.0298
label_smoothing_0.1           1.0       1.0     0.0724     0.9339              NaN             -0.0661
test accuracy 1.0000 -> 1.0000 (+0.0000) | overconfidence gap -0.0298 -> -0.0661 (-0.0363)
```

- **Accuracy is unchanged at 1.000**, on validation and test. Both runs keep their epoch-2
  checkpoint, and the two validation curves overlap.
- **Confidence drops** from 0.970 to 0.934, and the histogram shows the mechanism: the baseline
  piles ~140 of 180 test images into the top bin (0.96–1.0), while label smoothing spreads them
  down into the 0.85–0.96 range — toward its 0.917 target.
- **Test loss rises** from 0.032 to 0.072. Plain cross-entropy penalizes every correct
  prediction made with less than full confidence, so a less-confident model scores worse under
  it even though it makes no more errors.
- `mean_confidence_wrong` is `NaN` for both, because neither model makes a single test error.

### What the gap metric can and cannot show here

With accuracy exactly 1.0, `overconfidence_gap = confidence − 1`, which can **never** be positive.
On this test set the metric cannot detect over-confidence at all; it reports how far below 1.0
the mean confidence sits. Both values are negative, and smoothing makes the gap more negative.
What that means for calibration is for the extension conclusion to decide.

---

## Two concepts examiners love to probe

1. **Why does label smoothing cap confidence near `1 − ε + ε/K`?** Cross-entropy against a
   soft target is minimized when the predicted distribution equals the target. Pushing the true
   class above 0.917 would push the other five below 0.0167 each, moving the prediction *away*
   from the target and increasing the loss.
2. **Why is the baseline run inside the same cell instead of reusing `FINAL_RUN`?** A controlled
   comparison needs both conditions produced by the same code path, in the same session, with
   the same seed, so that the smoothing argument is provably the only difference. Re-running the
   baseline costs well under a minute and, by reproducing `FINAL_RUN`'s numbers exactly, also shows the
   pipeline is deterministic.
