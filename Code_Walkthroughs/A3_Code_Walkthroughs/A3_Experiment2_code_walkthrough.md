# A3 Experiment 2 — Code Walkthrough

_Study note (not part of the graded submission). Explains the Experiment 2 notebook cell
(`40a8f465`). No new graded function — it runs `run_training` twice, switching between
`freeze_backbone` and `unfreeze_all` (Task 3) through the harness's `mode` argument._

---

## Key background: what fine-tuning buys, and what it costs

Frozen transfer trains a linear head on fixed features. Full fine-tuning lets gradient flow
into the backbone too, so its features can change to suit the task. The trade-off:

- **Capacity.** Fine-tuning EfficientNet-B0 trains 4,015,234 numbers instead of 7,686 —
  **522×** more. More capacity can fit the task better, or fit the training set's noise
  (Task 6 Failure A is this second outcome, made deliberate).
- **Compute.** A frozen backbone needs no backward pass through its layers, and no optimizer
  state for them. Fine-tuning pays for both.
- **Stability.** Large updates to pretrained weights can destroy useful features
  ("catastrophic forgetting"). This is why the harness uses a **10× smaller** learning rate
  for fine-tuning (`adam: 1e-4` instead of `1e-3`).

The comparison is only interpretable if both conditions start from the **same** pretrained
initialization, see the same data in the same order, and are evaluated the same way.

---

## The notebook cell logic (`40a8f465`)

```python
# Controlled: BEST_ARCHITECTURE, same local ImageNet initialization, same loaders/seed/epochs/optimizer.
for mode in ("frozen", "finetune"):
    run = run_training(f"exp2_{mode}", architecture=BEST_ARCHITECTURE, mode=mode, augmentation="none",
                       optimizer_name="adam", epochs=EPOCHS, seed=SEED)
    record_run(2, "frozen transfer vs full fine-tuning", run)

EXP2_CHOICE = pick_best(EXP2_RUNS)
BEST_MODE = RUNS[EXP2_CHOICE]["mode"]
print(f"trainable-parameter ratio finetune/frozen: ...x | cost ratio ...x per epoch")
```

- **What changes:** `mode`, and with it the learning rate the table assigns (`1e-3` frozen,
  `1e-4` fine-tune). The learning rate is part of the condition by design — using the same
  rate for both would make one of them badly tuned.
- **Same initialization:** each call builds a fresh `efficientnet_b0` from the same weight file
  and, after `seed_everything(SEED)`, the same random head.
- `exp2_frozen` is the same configuration as `exp1_efficientnet_b0`. It is re-run rather than
  copied, and reproduces the same numbers (best val 1.0000 at epoch 2, val loss 0.1633), which
  doubles as a determinism check across cells.
- The two ratios printed at the end are the computational evidence the task asks for.

---

## What it produced

```
          run  mode      lr      best_val_acc  best_val_loss  final_train_acc  trainable  sec/epoch
  exp2_frozen  frozen    0.0010  1.0           0.1633         0.9986           7686       3.2711
exp2_finetune  finetune  0.0001  1.0           0.5908         1.0000           4015234    7.9038

trainable-parameter ratio finetune/frozen: 522.4x | cost ratio 2.42x per epoch
validation accuracy saturates at 1.0000 for every condition, so the tie-break applies:
selected 'exp2_frozen' on the lowest validation loss (0.1633)
BEST_MODE = frozen
```

- Both conditions reach 100 % validation accuracy, so accuracy cannot separate them.
- **522× the trainable parameters cost only 2.42× the time per epoch.** The forward pass
  through the backbone dominates both conditions; the extra backward pass and optimizer
  update roughly double-and-a-half it.

### What the tie-break actually compared

The fine-tuned run reached 1.000 at **epoch 1**, so — because `run_training` keeps the
*earliest* epoch that reaches the best accuracy — its kept checkpoint is the epoch-1 model,
whose validation loss is still 0.5908. The frozen run reached 1.000 at epoch 2, with loss
0.1633. The tie-break compared those two checkpoints.

The Exp 2 panel of `training_curves.png` shows the rest of the story: the fine-tuned run's
validation loss keeps falling after epoch 1, to roughly 0.06 at epoch 2 and close to zero by
epoch 4, ending well below the frozen run's (~0.03 at epoch 10). Its kept checkpoint is not
the point where it was at its best by loss — it is just the first epoch at 100 % accuracy.

So "frozen was selected" is a statement about the two **kept checkpoints** under this
selection rule. Whether it means more than that is a question for the analysis.

---

## Two concepts examiners love to probe

1. **Why is the trainable-parameter ratio (522×) so much larger than the time ratio (2.42×)?**
   Cost scales with the work done per image, not with the number of trainable weights. Both
   conditions pay a full forward pass through the backbone. Frozen training then back-propagates
   only through the tiny head: no backbone parameter requires a gradient, so autograd stops
   there. Fine-tuning back-propagates through the whole backbone, computing both activation and
   weight gradients — roughly twice the forward cost — plus an optimizer update. Forward + full
   backward is about 3× a forward pass, which is in line with the measured 2.42× (2.56× in an earlier run of the notebook).
2. **Why give fine-tuning a smaller learning rate?** The backbone's weights already encode
   useful ImageNet features. A step size suitable for a randomly initialized head would move
   them far from that solution in the first few batches, discarding what transfer learning is
   supposed to reuse.
