# A3 Experiment 3 — Code Walkthrough

_Study note (not part of the graded submission). Explains the Experiment 3 notebook cell
(`469368cb`). No new graded function — it uses `training_transform` and `AUGMENTATION_LEVELS`
from Task 1 to draw the examples figure, then `run_training` once per level._

---

## Key background: augmentation is a claim about invariance

Augmenting training data tells the model "this changed image has the **same label**." That is
only true for transformations the classes are actually invariant to:

- **Flips** preserve every class here: a mirrored horizontal stripe is still horizontal, a
  mirrored `/` diagonal is a `\` diagonal, which is still "diagonal".
- **Colour jitter** preserves every class: these patterns are defined by geometry, not colour.
- **Rotation** does **not** preserve every class. Rotating vertical stripes by 90° makes
  horizontal stripes; rotating by 30° moves them toward "diagonal". The orientation classes are
  defined by exactly what rotation changes.
- **Random-resized crops** change the apparent frequency of a pattern; mostly harmless here.

The three levels from Task 1 step from safe to risky: `none` (resize only), `moderate`
(flips + mild brightness/contrast), `strong` (crop + flips + ±30° rotation + strong colour
jitter + 10 % grayscale). The validation and test transforms are identical in all three, which
is what makes the comparison controlled.

---

## The notebook cell logic (`469368cb`)

### The examples figure

```python
archive = np.load(a3_tools.dataset_path())
examples = [int(np.flatnonzero(archive["train_labels"] == index)[0]) for index in range(NUM_CLASSES)]
rows = [("original", None)] + [(level, sc.training_transform(level)) for level in sc.AUGMENTATION_LEVELS]
a3_tools.seed_everything(SEED)
for row, (label, transform) in enumerate(rows):
    for column, index in enumerate(examples):
        image = Image.fromarray(archive["train_images"][index])
        shown = np.asarray(image) / 255.0 if transform is None else denormalize(transform(image))
```

- **One example per class**: `np.flatnonzero(labels == index)[0]` is the first training image
  of each class, so every column is a different class and the figure covers all six.
- **The same image goes through every level.** Each column is one source image; the rows differ
  only in transform. That makes the figure itself a controlled comparison.
- The **original** row is the raw 64×64 `uint8` image scaled to `[0, 1]`. The other rows are
  real transformed training tensors (224×224, ImageNet-normalized) passed through
  `denormalize` so they display correctly. Black corners in the `strong` row are the empty
  regions `RandomRotation` fills with zeros.
- `seed_everything(SEED)` before drawing makes the random rows reproducible: re-running the
  cell draws the same flips, crops, and angles.
- The figure is saved and closed (`plt.close`) so it is not rendered inline.

### The controlled runs

```python
for level in sc.AUGMENTATION_LEVELS:
    run = run_training(f"exp3_{level}", architecture=BEST_ARCHITECTURE, mode=BEST_MODE, augmentation=level,
                       optimizer_name="adam", epochs=EPOCHS, seed=SEED)
    record_run(3, "training augmentation strength", run)
BEST_AUGMENTATION = RUNS[pick_best(EXP3_RUNS)]["augmentation"]
```

- **What changes:** only `augmentation`. Architecture and mode come from Experiments 1–2
  (`efficientnet_b0`, `frozen`); the optimizer stays Adam because Experiment 4 has not run yet.
- `run_training` builds new loaders for each level through `build_dataloaders`, so the level is
  applied by the graded function, not by notebook code.
- `exp3_none` is again the same configuration as `exp1_efficientnet_b0` and `exp2_frozen`, and
  reproduces their numbers.

---

## What it produced

```
          run  augmentation  best_val_acc  best_val_loss  final_train_acc     gap  sec/epoch
    exp3_none  none          1.0           0.1633         0.9986          -0.0014     3.0532
exp3_moderate  moderate      1.0           0.1969         0.9972          -0.0028     3.0480
  exp3_strong  strong        1.0           0.3204         0.9792          -0.0208     3.2521

validation accuracy saturates at 1.0000 for every condition, so the tie-break applies:
selected 'exp3_none' on the lowest validation loss (0.1633)
BEST_AUGMENTATION = none
```

- All three reach 100 % validation accuracy; the loss tie-break decides.
- **Strong augmentation slows learning.** It reaches 1.000 at epoch 5 instead of epoch 2, its
  kept-epoch validation loss is twice as high, and its training accuracy finishes lowest
  (0.979). In the Exp 3 panel of `training_curves.png` the strong run's training accuracy
  stays visibly below the others for the whole run.
- The gaps are negative for every level: training accuracy is a running average over
  augmented batches in training mode, while validation is a clean pass over un-augmented
  images (see the Task 3 walkthrough).
- Strong augmentation costs ~7 % more time per epoch (14 % in an earlier, less loaded run): the crop, rotation, and colour
  transforms run on the CPU for every image.
- The 30° rotation in `strong` is a mild version of the failure Task 6 Failure B constructs on
  purpose with 90° rotations.

---

## Two concepts examiners love to probe

1. **Why must augmentation never touch validation or test?** Validation and test estimate
   performance on real inputs. Augmenting them would measure performance on a distribution the
   model will never be deployed on, and — because the transforms are random — would make the
   measurement noisy. Task 1's transform dictionary makes this structurally impossible.
2. **How can augmentation hurt?** When a transform changes the label but the training target
   does not change with it. Rotating a "vertical" image into horizontal stripes and still
   labelling it "vertical" teaches the model that the two classes look alike. Augmentation is
   only regularization when it is label-preserving.
