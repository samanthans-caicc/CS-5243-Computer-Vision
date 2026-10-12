# A3 Experiment 7 — Code Walkthrough

_Study note (not part of the graded submission). Explains the Experiment 7 notebook cell
(`60188424`): the Gaussian-noise robustness sweep, `robustness_results.png`, and the saving of
`trained_model.pt` and `model_metadata.json`. It reuses `ArrayImageDataset`,
`evaluation_transform` (Task 1), `evaluate_model` (Task 3), and `classification_metrics`
(Task 4)._

---

## Key background: robustness is a separate question from accuracy

A test set drawn from the same distribution as training measures *in-distribution*
performance. Real inputs are rarely that clean — sensors add noise, compression adds
artifacts. A **robustness test** applies one controlled corruption to the test images,
**without retraining**, and measures how performance degrades as the corruption grows.

"Controlled" has the same meaning as in Experiments 1–4: one corruption type, a range of
strengths, the same model, the same images, the same evaluation. Only the corruption strength
changes from point to point.

Using test data here is allowed — the README says so — because nothing is selected from it:
the model is already fixed, and the sweep only describes it.

---

## The notebook cell logic (`60188424`)

### The degradation

```python
NOISE_SIGMAS = [0.0, 0.02, 0.05, 0.10, 0.20]
noise_rng = np.random.default_rng(SEED)
clean_test = archive["test_images"].astype(np.float32)

def noisy_test_loader(sigma):
    noisy = np.clip(clean_test + noise_rng.normal(0.0, sigma * 255.0, clean_test.shape), 0, 255).astype(np.uint8)
    dataset = sc.ArrayImageDataset(noisy, test_labels, sc.evaluation_transform(), CLASS_NAMES)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS), noisy
```

- **σ is a fraction of the 0–255 range.** σ = 0.10 means a per-pixel, per-channel standard
  deviation of 25.5 grey levels.
- **Noise is added in pixel space, before the transform.** The noisy `uint8` images then go
  through the *same* `evaluation_transform` as the clean test set — resize to 224, ImageNet
  normalization. The model sees noise the way a camera would deliver it.
- Two side effects worth knowing:
  - Noise is drawn at **64×64** and then bilinearly upsampled to 224, so each noise sample
    becomes a smooth ~3.5-pixel blob rather than single-pixel speckle.
  - **Clipping** to `[0, 255]` truncates the noise near black and white, so near saturated
    pixels it is no longer zero-mean.
- σ = 0 still calls `normal(0, 0, …)`, which returns zeros, so the first point is exactly the
  clean test set — the sweep includes its own baseline.
- One seeded generator is shared across σ values in order, so the whole sweep is reproducible
  (it was reproduced exactly outside the notebook from the saved checkpoint).

### The sweep and the records

```python
for sigma in NOISE_SIGMAS:
    loader, noisy_images = noisy_test_loader(sigma)
    evaluation = sc.evaluate_model(FINAL_RUN["model"], loader, nn.CrossEntropyLoss(), DEVICE)
    metrics = sc.classification_metrics(evaluation["labels"], evaluation["predictions"], CLASS_NAMES)
    degraded_examples[sigma] = noisy_images[0]
    robustness.append({..., "mean_confidence": float(evaluation["probabilities"].max(axis=1).mean())})
```

- `FINAL_RUN["model"]` is the validation-selected model, unchanged — no retraining, no
  fine-tuning on noisy data.
- Four metrics per σ (accuracy, loss, macro-F1, mean confidence) plus one `accuracy_drop`
  record: 21 rows for Experiment 7.
- **Mean confidence** is tracked next to accuracy because the two can come apart: a model can
  stay confident while it becomes wrong.
- `degraded_examples` keeps the first test image (a diagonal) at each σ for the figure.

### `robustness_results.png`

A line plot of accuracy and mean confidence against σ, with a dotted **chance line** at
1/6 = 0.167, next to a strip of the same test image at each σ with its accuracy in the title.
The curve shows the numbers; the strip shows what those σ values look like.

### Saving the submitted model

```python
checkpoint = a3_tools.save_state_dict(PATHS["models"] / "trained_model.pt", FINAL_RUN["model"])
model_metadata = {"architecture": ..., "num_classes": 6, "class_names": CLASS_NAMES,
                  "checkpoint": "outputs/models/trained_model.pt",
                  "parameter_counts": {"total": 4015234, "trainable": 7686},
                  "seed": SEED, "selected_by": SELECTED_BY, "training": {...},
                  "validation_accuracy": ..., "test_accuracy": ...}
assert size_mb <= 25.0
```

- `save_state_dict` saves `model.state_dict()` only — weights and BatchNorm buffers, no
  optimizer state — so it can be loaded with `build_model` + `load_state_dict`.
- 4,015,234 float32 parameters are 16.06 MB; the BatchNorm buffers and serialization overhead
  bring the file to **16.4 MB**,
  under the 25 MB limit (and above the validator's 10 MB *warning* threshold, which is why the
  validator lists it).
- `trainable: 7686` records that the model was trained frozen. The extra `training` block
  (mode, optimizer, rate, selected epoch, batch and image size) goes beyond the required schema,
  which the README allows.
- The `assert` makes the size limit a hard failure of the notebook rather than a validator
  surprise.

---

## What it produced

```
 sigma  accuracy   loss  macro_f1  mean_confidence
  0.00    1.0000 0.0322    1.0000           0.9702
  0.02    1.0000 0.0594    1.0000           0.9452
  0.05    0.9833 0.2740    0.9832           0.7890
  0.10    0.6111 1.1326    0.5893           0.4742
  0.20    0.1056 1.9242    0.0905           0.2627
original 1.0000 -> degraded (sigma=0.20) 0.1056 | drop 0.8944
```

- At σ ≤ 0.02 accuracy is untouched, but **confidence and loss already move** (loss nearly
  doubles). At σ = 0.05, accuracy loses three images while mean confidence falls to 0.79.
- Between σ = 0.05 and 0.10 accuracy falls from 0.98 to 0.61; at σ = 0.20 it is **0.106 — below
  chance** (0.167).

**Below chance means the predictions collapse onto a few classes.** Re-running the sweep from
the saved checkpoint (same seed, same order) reproduces the notebook's numbers exactly and gives
the predicted-class counts out of 180 (30 true images per class):

| σ | checker | circles | diagonal | dots | horizontal | vertical |
|---|---|---|---|---|---|---|
| 0.10 | 18 | 36 | 11 | **66** | 13 | 36 |
| 0.20 | 43 | 2 | 21 | **73** | **0** | 41 |

At σ = 0.20, "dots" receives 73 predictions and "horizontal" none; a uniform random guesser
would score 0.167, but a model that sends most images to a few wrong classes scores lower.

---

## Two concepts examiners love to probe

1. **Why can accuracy fall below chance?** Chance (1/K) is the score of guessing uniformly at
   random. A model that systematically maps corrupted inputs to the *same* few classes is worse
   than random for every class it never predicts. Below-chance accuracy is a sign of a
   systematic bias, not of random confusion.
2. **Why measure confidence alongside accuracy?** Accuracy only changes when the top class
   flips; confidence and loss respond as soon as the inputs move away from the training
   distribution. Here they move at σ = 0.02 while accuracy is still perfect — an early warning
   that accuracy alone would miss.
