# A3 Task 6 — Code Walkthrough

_Study note (not part of the graded submission). Explains the Task 6 notebook cell
(`1052d27f`): two constructed failures — label-noise memorization and a label-destroying
augmentation — each with a mitigation, and `failure_analysis.png`. No new graded function; it
reuses `build_dataloaders` and `ArrayImageDataset` (Task 1), the harness's `run_training` and
`test_report`, and `draw_confusion`._

---

## Key background: a failure analysis is an experiment, not an anecdote

The task asks for four things per failure: **evidence**, **mechanism**, **mitigation**,
**verification**. That is the structure of a controlled experiment: cause the failure on
purpose, measure it, change the thing you believe causes it, and measure again under otherwise
identical conditions. Because the bundled dataset is too easy to fail naturally (Experiments
1–6 all hit 100 %), both failures here are **constructed**: the cell injects a known cause so
the mechanism is known by design.

- **Failure A — label noise → memorization (overfitting).** Corrupt a share of the *training*
  labels. A model with enough capacity can fit those wrong labels image by image instead of
  learning the pattern, and validation (with clean labels) suffers.
- **Failure B — augmentation mismatch.** Train with 90° rotations. Rotation turns horizontal
  stripes into vertical ones while the label stays the same, so the training data itself says
  the two classes look alike.

---

## Failure A — the code

```python
LABEL_NOISE, FAILURE_EPOCHS = 0.30, 15

def noisy_label_loaders(fraction, augmentation, seed):
    base = sc.build_dataloaders(a3_tools.dataset_path(), BATCH_SIZE, augmentation, seed, NUM_WORKERS)
    clean = base["train"].dataset.labels.copy()
    generator = np.random.default_rng(seed)
    chosen = generator.choice(len(clean), int(round(fraction * len(clean))), replace=False)
    noisy = clean.copy()
    noisy[chosen] = generator.integers(0, NUM_CLASSES, len(chosen))
    dataset = sc.ArrayImageDataset(archive["train_images"], noisy, base["train"].dataset.transform, CLASS_NAMES)
    base["train"] = DataLoader(dataset, ..., shuffle=True, generator=torch.Generator().manual_seed(seed))
    return base, float((noisy != clean).mean())
```

- Starts from the graded loaders, so validation and test are untouched (clean labels,
  deterministic transforms), and replaces only the **training** loader.
- 216 of 720 images (30 %) are **reassigned** a uniformly random class. One time in six the
  random class equals the true one, so the share actually **changed** is lower: 177 / 720 =
  **24.6 %**, which the function returns and the cell prints.
- The seeded `default_rng` makes the corruption identical in both runs.

```python
FAILURE_A = run_training("failA_label_noise_overfit", architecture=BEST_ARCHITECTURE, mode="finetune",
                         augmentation="none", optimizer_name="adam", lr=1e-3, epochs=FAILURE_EPOCHS, ...)
FAILURE_A_FIX = run_training("failA_mitigated", architecture=BEST_ARCHITECTURE, mode="frozen",
                             augmentation="moderate", optimizer_name="adam", lr=1e-3, epochs=FAILURE_EPOCHS, ...)
```

- **Before:** full fine-tuning (4,015,234 trainable weights) with Adam at `1e-3` — 10× the
  table's fine-tuning rate — for 15 epochs. All three choices push toward memorization.
- **After:** frozen backbone (7,686 trainable) *and* moderate augmentation. The mitigation
  changes **two** levers at once; the verification shows the pair works, not which one did it.

### The memorization ceiling

```python
NOISE_CEILING = 1 - actual_noise   # 0.7542
```

A model that learns only the true pattern, and never memorizes a corrupted label, predicts the
true class for every image. It gets every **unchanged** label right and every **changed** label
wrong (by definition the changed label is not the true class). Its training accuracy is
therefore `1 − ν`, with `ν` the changed fraction. Anything above that can only come from fitting
corrupted labels one image at a time.

**An earlier version of this line was wrong.** It used `(1 − ν) + ν/K`, which printed 0.7951.
That formula is right only when `ν` is the fraction *reassigned* — some reassignments
coincidentally keep the true label, and those are the `ν/K`. But `actual_noise` is the fraction
*changed*, which already excludes the coincidences, so the extra term counted them twice:

| Formula | Value |
|---|---|
| `1 − actual_noise` (changed fraction) — **used now** | **0.7542** |
| `(1 − 0.30) + 0.30/6` (reassigned fraction, expected value) | 0.7500 |
| `(1 − actual_noise) + actual_noise/6` — the earlier, double-counting version | 0.7951 |

The correction moves the printed "vs ceiling" values, the dashed line in
`failure_analysis.png`, and the `memorization_free_train_ceiling` row in the CSV.

### What Failure A produced

```
memorization-free training-accuracy ceiling ~ 0.7542
before: train 0.9222 (+0.1681 vs ceiling) / val 0.8556 -> gap +0.0667 | clean test 0.9889 | trainable 4,015,234
after : train 0.7472 (-0.0069 vs ceiling) / val 0.9944 -> gap -0.2472 | clean test 1.0000 | trainable 7,686
verified change: validation accuracy +0.1389, clean test accuracy +0.0111, train-validation gap -0.3139
```

- The **before** curves (top-left panel): validation accuracy starts at 0.983 in epoch 1 and
  falls — to 0.767 at epoch 8 and 0.717 at epoch 11 — while training accuracy climbs from 0.68
  past the ceiling to 0.922 at epoch 15. Training accuracy rising above the ceiling while clean
  validation accuracy falls is the signature of memorization.
- The **after** curves (top-right): training accuracy sits flat at 0.74–0.76 — on the ceiling —
  and validation stays at 0.994–1.000 for all 15 epochs.
- **These before-mitigation values are not bit-reproducible.** An earlier run of the same
  notebook gave train 0.9153 / val 0.8778 for the fine-tuned run; this run gives 0.9222 /
  0.8556. Full fine-tuning on a GPU is not deterministic under `seed_everything` alone (see the
  Experiment 9 walkthrough for the mechanism). The frozen mitigated run reproduced exactly, and
  so did both Failure B runs. The qualitative picture — training above the ceiling, validation
  falling — is the same in both runs.

**Which model the "clean test" numbers describe.** The printed train/val values are
**final-epoch** values, but `test_report` evaluates the **restored best-validation
checkpoint**. For the before run that is **epoch 1** (val 0.983), *before* memorization set
in, so its test 0.9889 is the test accuracy of an almost un-memorized model. The epoch-15
memorizing model's test accuracy was never measured; its final validation accuracy (0.856) is
the measured evidence of the damage. The `+0.0111` "clean test accuracy gain" therefore
compares the mitigated model with the *epoch-1* before-model, not with the model in the
before curves.

---

## Failure B — the code

```python
ROTATION_TRANSFORM = transforms.Compose([
    transforms.Resize((sc.IMAGE_SIZE, sc.IMAGE_SIZE)),
    transforms.RandomChoice([transforms.RandomRotation((angle, angle)) for angle in (0, 90, 180, 270)]),
    transforms.ToTensor(), transforms.Normalize(sc.IMAGENET_MEAN, sc.IMAGENET_STD)])
```

- `RandomRotation((a, a))` with a degenerate range rotates by **exactly** `a`; `RandomChoice`
  picks one of the four angles per image. On a square image, 90° rotations are exact — no
  black corners.
- **Who is affected:** 0° and 180° leave every class unchanged. 90° and 270° swap horizontal
  and vertical stripes, while checkers, circles, dots, and diagonals (`/` ↔ `\`) stay in their
  class. Half the horizontal and vertical training images are therefore shown with the *other*
  class's appearance and their *own* label.

```python
FAILURE_B     = run_training("failB_rotation_mismatch", ..., mode="finetune", augmentation="rot90",
                             optimizer_name="adam", epochs=8, loaders=rotated_loaders(SEED), ...)
FAILURE_B_FIX = run_training("failB_mitigated", ..., mode="finetune", augmentation="moderate",
                             optimizer_name="adam", epochs=8, ...)
```

- Fine-tuned (Adam `1e-4`) for 8 epochs in both. The `"rot90"` string is just a label for the
  run table; the actual transform comes from `rotated_loaders`.
- The mitigation swaps the rotation for the graded **moderate** level — so it removes rotation
  *and* adds flips and colour jitter. Flips are label-preserving here, so the change that
  matters is the removed rotation.
- `swap(matrix)` counts horizontal→vertical plus vertical→horizontal errors in the test
  confusion matrix — the single number the mechanism predicts.

### What Failure B produced

```
before: test accuracy 0.9389 | horizontal/vertical accuracy 0.8167 | mutual confusions 11
after : test accuracy 1.0000 | horizontal/vertical accuracy 1.0000 | mutual confusions 0
```

- All 11 errors are horizontal↔vertical (5 one way, 6 the other); every other class is 1.00.
  The confusion matrix shows exactly the pattern the mechanism predicts and nothing else.
- Horizontal/vertical accuracy is 0.82, not 0.50. If orientation were the *only* difference,
  the rotated training data would make the two classes indistinguishable; above-chance
  accuracy suggests the model also used cues rotation does not change (stripe width or colour
  statistics, for example). That is an inference — the cell does not test it.
- These are best-validation checkpoints too (before: epoch 4, val 0.911).

---

## The records and the figure

Each failure writes six metrics for each of its two stages, plus derived values (corrupted
fraction, ceiling, gap reduction, test-accuracy gain, oriented-class gain, confusions before
and after) — 31 `task6` rows. `failure_analysis.png` is a 2×2 grid: Failure A's train and
validation curves before and after, with the shaded train−validation gap and the dashed
ceiling; Failure B's row-normalized test confusion matrices before and after.

---

## Two concepts examiners love to probe

1. **Why does training accuracy above `1 − ν` prove memorization?** Corrupted labels are random
   with respect to the image. No rule based on what the image looks like can predict them, so
   the only way to get a corrupted training image "right" is to have stored that specific
   image–label pair. Every point above `1 − ν` is a memorized example.
2. **Why does freezing the backbone stop it?** A linear head on fixed features can only draw
   six linear boundaries. It cannot carve out exceptions for 177 individual images without
   misclassifying their neighbours, so the best it can do is predict the true pattern — which
   puts training accuracy at the ceiling and leaves validation intact.
