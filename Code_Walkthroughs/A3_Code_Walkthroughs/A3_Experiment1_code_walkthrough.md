# A3 Experiment 1 — Code Walkthrough

_Study note (not part of the graded submission). Explains the Experiment 1 notebook cell
(`68360a89`). No new graded function — it calls `run_training`, `record_run`,
`inference_milliseconds`, `summarize`, `pick_best`, and `selection_note` from the harness
(see `A3_Experiments_harness_code_walkthrough.md`), which in turn call `build_dataloaders`,
`build_model`, `freeze_backbone`, `train_one_epoch`, and `evaluate_model`._

---

## Key background: two efficient architectures, two different bargains

Both candidates were designed for phones, but they spend their budget differently:

- **MobileNetV3-Small** (1.52 M parameters) — depthwise-separable convolutions,
  squeeze-and-excitation blocks, and hard-swish activations, sized as small as possible. Its
  backbone outputs 576 channels, and its head keeps an extra pretrained `Linear(576, 1024)`
  before the class layer.
- **EfficientNet-B0** (4.02 M parameters) — MBConv blocks with depth, width, and input
  resolution scaled together. It outputs 1280 pooled features directly into the class layer.

An architecture comparison has to answer two questions, not one: *which learns the task
better?* and *what does it cost?* — in parameters, training time, and inference latency. The
task asks for all three cost measurements in the metrics file.

To compare architectures fairly in a **transfer** setting, both are run frozen: only the head
learns, so the comparison is about the quality of each backbone's pretrained features.

---

## The notebook cell logic (`68360a89`)

```python
# Controlled: identical seed, loaders, augmentation=none, Adam, frozen backbone, and epoch budget.
for architecture in sc.SUPPORTED_ARCHITECTURES:
    run = run_training(f"exp1_{architecture}", architecture=architecture, mode="frozen",
                       augmentation="none", optimizer_name="adam", epochs=EPOCHS, seed=SEED)
    latency = inference_milliseconds(run["model"], run["loaders"]["test"])
    record_run(1, "frozen transfer (architecture comparison)", run,
               extra={"inference_latency": (latency, "ms/image")})

EXP1_RUNS = [f"exp1_{architecture}" for architecture in sc.SUPPORTED_ARCHITECTURES]
EXP1_CHOICE = pick_best(EXP1_RUNS)
BEST_ARCHITECTURE = RUNS[EXP1_CHOICE]["architecture"]
```

- **What changes:** only `architecture`. Mode (`frozen`), augmentation (`none`), optimizer
  (`adam`, `lr = 1e-3` from the table), epochs (10), seed (5243), and loaders are identical.
  Because `run_training` builds the loaders from the same seed, both models see the same
  batches in the same order.
- **Latency is measured on the test loader's *images*.** That is not a selection decision: no
  labels are read and nothing is chosen from it. The test images are just a fixed, realistic
  batch set to time forward passes on. The selection uses only validation numbers.
- `record_run` writes the standard ten-metric block (best validation accuracy and loss, final
  train accuracy and loss, gap, selected epoch, total/trainable parameters, wall time,
  seconds per epoch) plus latency — 11 rows per architecture, 22 for the experiment.
- `BEST_ARCHITECTURE` becomes a global that **every later experiment reads**. Experiments 2–9,
  Task 6, and Task 7 all use the architecture picked here.

---

## What it produced

| | MobileNetV3-Small | EfficientNet-B0 |
|---|---|---|
| best validation accuracy | 0.9889 (epoch 7) | **1.0000** (epoch 2) |
| validation loss at kept epoch | 0.1987 | 0.1633 |
| final train accuracy | 0.9986 | 0.9986 |
| total / trainable parameters | 1,524,006 / 6,150 | 4,015,234 / 7,686 |
| seconds per epoch | 1.56 | 2.45 |
| inference latency | **0.49 ms/image** | 1.82 ms/image |

```
selected 'exp1_efficientnet_b0' on the highest validation accuracy (1.0000)
BEST_ARCHITECTURE = efficientnet_b0
```

- This is the **one** selection in Experiments 1–4 decided by accuracy rather than by the loss
  tie-break. MobileNet's 0.9889 is 178 of 180 validation images — two errors.
- EfficientNet costs about **1.6×** the training time per epoch and **3.7×** the inference
  latency, with 2.6× the parameters. The trainable counts differ much less (7,686 vs 6,150)
  because a frozen run only trains the head, whose size depends on the feature width
  (1280 vs 1024), not on the backbone size. Timings depend on machine load (they rose ~30 % between two runs of this notebook); the ratios are the stable part.
- In the training-curves figure (Exp 1 panel), EfficientNet's validation loss drops to ~0.16
  after two epochs while MobileNet's descends slowly over all ten epochs and is still ~0.14 at
  the end. MobileNet's validation accuracy climbs from 0.83 to 0.99 over the run.

---

## Why the epoch-7 vs epoch-2 difference matters later

`run_training` keeps the earliest epoch that reached the best validation accuracy.
EfficientNet hit 1.000 at epoch 2 and **stayed there**, so every later EfficientNet run in
Experiments 2–4 is also represented by an early checkpoint. From Experiment 2 onward every
condition ties at 1.000 validation accuracy, and decisions fall to the validation-loss
tie-break (see the harness walkthrough). Experiment 1 is where that saturation starts: the
dataset is easy enough for EfficientNet's frozen features to solve it in two epochs.

---

## Two concepts examiners love to probe

1. **Why compare architectures with a frozen backbone instead of fine-tuning both?** Frozen
   transfer isolates the pretrained representation: the only learned part is a linear head, so
   a difference in accuracy reflects how linearly separable each backbone makes the classes.
   Fine-tuning would mix that with how well each backbone adapts at a given learning rate.
2. **Why report latency separately from training time?** Training time per epoch includes the
   backward pass and data loading and scales with dataset size; latency is the per-image cost
   of using the model. A deployment decision depends on latency, and the two do not scale
   together — EfficientNet is 1.6× slower to train an epoch here but 3.7× slower to run.
