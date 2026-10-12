# A3 Experiment 9 — Code Walkthrough

_Study note (not part of the graded submission). Explains the Experiment 9 notebook cell
(`a1e5f004`): EuroSAT fine-tuning, the frozen linear-probe comparison, and
`eurosat_transfer_results.png`. It reuses `build_external_dataloaders` unchanged (see the
Experiment 8 walkthrough) and the harness's `run_training` and `test_report`._

---

## Key background: domain shift, and how to measure it

ImageNet is ground-level photographs of objects — dogs, cars, mugs — photographed from the
side, centred, at a natural scale. **EuroSAT** is Sentinel-2 satellite imagery: 64×64 overhead
tiles of land cover (crops, forest, rivers, highways), viewed from above, where "objects" are
textures and layouts rather than shapes. Transferring between them is a **domain shift**: the
input distribution changes, not just the label set.

The question is how much of the pretrained representation still works. One run cannot answer
that, so the cell trains **two**, on identical data:

- a **frozen linear probe** — the backbone exactly as ImageNet left it, only a new head
  trained. Its accuracy measures how linearly separable EuroSAT is *in ImageNet features*;
- a **fully fine-tuned** model — the backbone allowed to adapt.

The difference between them, the **adaptation gain**, is the part of the performance that
required changing the features to fit the new domain.

Unlike Skin Cancer, EuroSAT is roughly balanced (1,195 – 1,863 training images per class), so
this experiment isolates domain shift from imbalance.

---

## The notebook cell logic (`a1e5f004`)

```python
eurosat_loaders = sc.build_external_dataloaders("timm/eurosat-rgb", "label", BATCH_SIZE,
                                                BEST_AUGMENTATION, SEED, EXTERNAL_WORKERS)
EUROSAT_RUN = run_training("exp9_eurosat", architecture=BEST_ARCHITECTURE, mode="finetune", ...,
                           epochs=EXTERNAL_EPOCHS, loaders=eurosat_loaders, num_classes=10, ...)
EUROSAT_FROZEN = run_training("exp9_eurosat_frozen", architecture=BEST_ARCHITECTURE, mode="frozen", ...,
                              epochs=EXTERNAL_EPOCHS, loaders=eurosat_loaders, num_classes=10, ...)
print(f"... adaptation gain {eurosat_metrics['accuracy'] - frozen_metrics['accuracy']:+.4f}")
```

- **The same graded function, a different dataset.** `build_external_dataloaders` finds
  `label` is a `ClassLabel`, so the class names come from the feature and no string mapping is
  built. The published 16,200 / 5,400 / 5,400 splits are used as they are — no resplit, no
  carving.
- **The two runs share one loaders object**, so they train on the same images with the same
  head initialization (each `run_training` call reseeds the global RNG before building the
  model). Intended difference: `mode` and, with it, the table's learning rate — SGD `1e-3` for
  fine-tuning, `1e-2` for the frozen head.
- **One unintended difference: shuffle order.** The train loader's private `torch.Generator`
  is created once inside `build_external_dataloaders` and advances every epoch.
  `seed_everything` does not reset it. The fine-tuned run consumes the generator's first three
  epochs of shuffles; the frozen run, reusing the same loader, gets the *next* three. The batch
  orders differ between the two conditions. It is unlikely to matter at 16,200 images and
  three epochs, but strictly the comparison is not order-matched. Building the loaders twice
  (once per run), as Experiments 1–4 do through `run_training`, would remove it.
- Both use the 3-epoch external budget. The frozen run trains 12,810 parameters
  (1280 · 10 + 10); the fine-tuned run trains 4,020,358.
- Records: two 10-metric run blocks + 3 test metrics × 2 + 1 adaptation gain + 10 classes × 2 =
  47 Experiment 9 rows.

### `eurosat_transfer_results.png`

```python
raw_test_labels = np.array([int(value) for value in eurosat_loaders["test"].dataset.split["label"]])
for index in range(len(EUROSAT_CLASSES)):
    position = int(np.flatnonzero(raw_test_labels == index)[0])
    image, _ = eurosat_loaders["test"].dataset[position]
    tiles.append(denormalize(image))
```

- **Panel 1 — example tiles:** the first test image of each class, taken *through the test
  dataset* so it is exactly what the model saw (resized to 224 and normalized), then
  denormalized for display. Each title carries that class's fine-tuned test accuracy. Reading
  the label column directly avoids decoding every image just to find one per class.
- **Panel 2 — row-normalized confusion matrix** of the fine-tuned model (`draw_confusion` from
  the harness), so each row reads as per-class recall.
- **Panel 3 — per-class accuracy, frozen versus fine-tuned,** as paired horizontal bars with a
  chance line at 0.1. This panel is where the domain-adaptation effect is visible class by class.

---

## What it produced

```
[exp9_eurosat]        finetune  sgd lr=0.001  best val acc 0.9635 @ epoch 3  trainable 4,020,358
test accuracy 0.9674 | macro F1 0.9673 | chance 0.100
[exp9_eurosat_frozen] frozen    sgd lr=0.01   best val acc 0.9202 @ epoch 3  trainable 12,810
frozen linear probe test accuracy 0.9224 | fine-tuned 0.9674 | adaptation gain +0.0450
```

- **Frozen ImageNet features already reach 0.922** on satellite tiles — nine times chance — with
  only a linear head trained.
- **Fine-tuning adds 4.5 points**, to 0.967, at about 2.8× the training time (505 s vs 180 s).
- Accuracy and macro-F1 agree (0.9674 / 0.9673), as expected on balanced classes — the
  opposite of Experiment 8.
- Per class, fine-tuned accuracy ranges from 0.911 (*PermanentCrop*) to 0.996 (*Residential*).
  The confusion matrix shows PermanentCrop's errors going mostly to HerbaceousVegetation (6 %)
  and AnnualCrop (3 %), and Pasture's (0.93) spread over AnnualCrop, HerbaceousVegetation, and
  Forest — vegetation classes that look alike from above.
- In panel 3, the frozen probe is weakest on *Highway* and *PermanentCrop*, and the largest
  visible gain from fine-tuning is on *Highway*, a thin, linear structure rather than a texture.
  Fine-tuning does not help every class: on *Forest* the fine-tuned model (0.959) is slightly
  below the frozen probe.
- **Both runs selected epoch 3, the last epoch of the budget**, with validation still improving;
  the 3-epoch budget, not convergence, is what ended both runs.

### This experiment's numbers are not bit-reproducible

An earlier run of the same notebook, same seed, gave the fine-tuned model **0.9750** test
accuracy and a **+0.0526** gain; this run gives 0.9674 and +0.0450. The frozen probe reproduced
exactly (0.9224 both times). The difference comes from GPU non-determinism, not from the code:
`seed_everything` seeds every random generator, but it does not set
`torch.backends.cudnn.deterministic` or `torch.use_deterministic_algorithms(True)`, and cuDNN's
backward-pass kernels for convolutions can sum floating-point values in a different order from
run to run. A frozen run back-propagates only through one linear layer, so it is unaffected;
full fine-tuning back-propagates through the whole backbone, and over 1,500 steps the tiny
differences grow into a visible one. The same effect moved Task 6 Failure A; Experiments 2 and 8
(also fine-tuned) moved only in the fifth or sixth decimal place.

---

## Two concepts examiners love to probe

1. **Why is a frozen linear probe the right baseline for domain shift?** It holds the
   representation fixed at what ImageNet taught it. Any accuracy it reaches is transfer in the
   strict sense — features learned elsewhere, reused unchanged. The gap to fine-tuning measures
   how much the features themselves had to change for the new domain, which no single run can
   show.
2. **Why would ImageNet features transfer to satellite imagery at all?** The early and middle
   layers of a CNN learn edges, colour contrasts, and texture statistics, which are generic to
   natural images from any viewpoint. What does not transfer as well are the later, more
   object-specific features — and those are what fine-tuning adjusts.
