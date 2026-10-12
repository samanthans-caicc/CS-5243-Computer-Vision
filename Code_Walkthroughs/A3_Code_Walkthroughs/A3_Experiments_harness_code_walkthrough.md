# A3 Experiment Harness — Code Walkthrough

_Study note (not part of the graded submission). Explains the shared harness cell at the top
of Section 5 (`d6d725d1`): `record`, `write_metrics`, `make_optimizer`, `run_training`,
`record_run`, `summarize`, `pick_best`, `selection_note`, `inference_milliseconds`,
`test_report`, `per_class_table`, `denormalize`, `draw_confusion`, and `external_label_counts`.
No graded function lives here — every experiment, Task 6, and Task 7 calls into it._

---

## Key background: one routine, so only one thing can change

Experiments 1–4 each claim "everything else held fixed." That claim is only as good as the
code path. If each experiment had its own training loop, a different learning rate, epoch
count, or checkpoint rule could slip into one condition unnoticed. The harness puts **all**
training through one function, `run_training`, so two conditions differ in exactly the
keyword arguments the experiment passes — and nothing else.

It also enforces the split discipline: `run_training` never touches the test loader. Test data
is only read by `test_report`, which is called *after* a selection has been made.

---

## The metrics table: `record` and `write_metrics`

```python
METRIC_COLUMNS = ["experiment", "method", "condition", "metric", "value", "units"]
METRIC_ROWS, RUNS = [], {}

def record(experiment, method, condition, metric, value, units):
    METRIC_ROWS.append({... "value": float(value), "units": units})

def write_metrics():
    frame = pd.DataFrame(METRIC_ROWS, columns=METRIC_COLUMNS)
    frame.to_csv(METRICS_PATH, index=False)
    return frame
```

- **Long form**: one number per row, described by the other five columns. New experiments add
  rows, never columns, so the README schema holds for all 281 records.
- `float(value)` turns NumPy scalars and ints into plain floats so the CSV has one type.
- **"Cumulative" is implemented in memory.** `write_metrics` rewrites the whole file from
  `METRIC_ROWS` each time. A top-to-bottom run produces the file exactly once over. Re-running a
  single cell in a live kernel would *append that cell's rows again*, duplicating them — the
  reason the submission instructions say restart and run all.

---

## `make_optimizer` and the learning-rate table

```python
EPOCHS = 10
DEFAULT_LR = {"frozen": {"adam": 1e-3, "sgd": 1e-2}, "finetune": {"adam": 1e-4, "sgd": 1e-3}}

def make_optimizer(name, parameters, lr):
    if name == "adam": return torch.optim.Adam(parameters, lr=lr)
    if name == "sgd":  return torch.optim.SGD(parameters, lr=lr, momentum=0.9)
```

- A fresh head tolerates large steps; a pretrained backbone does not. Fine-tuning rates are
  **10× smaller** than frozen rates so the backbone's ImageNet features are adjusted, not
  overwritten.
- SGD gets a 10× larger rate than Adam, the conventional pairing: Adam rescales each step by a
  running gradient magnitude, SGD does not.
- An explicit `lr=` always overrides the table — Task 6 Failure A uses `lr=1e-3` for
  fine-tuning on purpose.

---

## `run_training` — the core

```python
a3_tools.seed_everything(seed)
if loaders is None:
    loaders = sc.build_dataloaders(a3_tools.dataset_path(), BATCH_SIZE, augmentation, seed, NUM_WORKERS)
model = sc.build_model(architecture, num_classes, a3_tools.weight_path(architecture))
sc.freeze_backbone(model) if mode == "frozen" else sc.unfreeze_all(model)
optimizer = make_optimizer(optimizer_name, [p for p in model.parameters() if p.requires_grad], learning_rate)
train_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
eval_criterion  = nn.CrossEntropyLoss()
```

- **Seed first.** Seeding before building means the new head's random initialization is the
  same for every condition with the same architecture and seed.
- **Fresh model every call**, built from the local ImageNet weights. No condition inherits
  weights from a previous run.
- Loaders can be passed in, which is how Experiments 8–9 and Task 6 reuse the same loop on
  Hugging Face data, noisy labels, or rotated images.
- **Two criteria.** Training may use label smoothing (Task 7); validation always uses plain
  cross-entropy, so validation losses are comparable across every run.

```python
for epoch in range(1, epochs + 1):
    train = sc.train_one_epoch(model, loaders["train"], optimizer, train_criterion, DEVICE)
    validation = sc.evaluate_model(model, loaders["validation"], eval_criterion, DEVICE)
    ...
    if validation["accuracy"] > best_val:
        best_val, best_epoch = validation["accuracy"], epoch
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
model.load_state_dict(best_state)
```

- **Best-validation checkpointing.** After every epoch the model is evaluated on validation;
  the best epoch's full `state_dict` (including BatchNorm buffers) is copied to the CPU. At the
  end, the model is **restored to that epoch**. Every later use of `run["model"]` — test
  evaluation, embeddings, the saved checkpoint — uses the best-validation weights, not the
  last epoch's.
- `.clone()` matters: `state_dict()` returns references to live tensors that keep training.
- **The comparison is strict (`>`).** Once validation accuracy hits its maximum, later epochs
  that only *tie* it do not replace the checkpoint. The kept model is the **earliest** epoch
  that reached the best accuracy. This becomes important below.

The returned `run` dictionary holds the history table, the restored model, the loaders, and
derived numbers. Two gaps are reported, and they are not the same thing:

- `generalization_gap` = final-epoch train accuracy − **best** validation accuracy.
- `final_gap` = final-epoch train accuracy − final-epoch validation accuracy (Task 6 uses this).

---

## Selection: `pick_best` and `selection_note`

```python
def pick_best(names):
    return min(names, key=lambda n: (-RUNS[n]["best_val_accuracy"], RUNS[n]["best_val_loss"]))
```

Sorting by the tuple `(−accuracy, loss)` means: highest validation accuracy wins; if tied,
lowest validation loss wins. Only validation numbers enter the key — test data cannot influence
a selection. `selection_note` prints which rule actually decided, so the notebook states
plainly when accuracy tied at 1.000 and the loss tie-break was used.

**What the tie-break compares.** `best_val_loss` is the validation loss *at the kept epoch*.
Combined with the strict `>` above, a run that reaches 100 % accuracy very early is
represented by an early, still-high-loss checkpoint. Experiment 2 is the clear case: the
fine-tuned run reached 1.000 at **epoch 1** (validation loss 0.5908) and kept that epoch; the
frozen run reached it at epoch 2 (loss 0.1633). The tie-break therefore chose frozen, even
though the training-curves figure shows the fine-tuned run's validation loss falling far below
the frozen run's in later epochs. The rule is consistent — it compares the models that would
actually be shipped — but it answers "which kept checkpoint has lower loss," not "which
condition reaches lower loss."

---

## Timing, test evaluation, and display helpers

```python
def inference_milliseconds(model, loader, repeats=5):
    batches = [images for images, _ in loader]          # pre-load: no disk / decode time
    with torch.no_grad():
        for _ in range(2): model(batches[0].to(DEVICE))   # warm-up
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    ... 5 passes over every batch ...
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / total
```

- Batches are loaded once, so the measurement excludes data loading. The host-to-GPU copy is
  still inside the timer.
- **Warm-up** absorbs one-time costs (CUDA kernel selection, memory allocation).
- **`synchronize`** is essential: CUDA calls return before the GPU finishes, so without it the
  timer would measure how fast work is *queued*, not done.

`test_report(run)` is the **only** place the test split is evaluated — once per selected run.
`per_class_table` turns the metrics dictionary into a DataFrame. `denormalize` reverses
ImageNet normalization (`x·std + mean`, clipped to `[0, 1]`) so transformed tensors can be
shown. `draw_confusion` is a multi-panel version of `plot_confusion_matrix` that draws into an
existing axis. `external_label_counts` counts classes by reading only the Hugging Face label
column — it never decodes an image, so counting 7,153 training labels is instant.

---

## Two concepts examiners love to probe

1. **Why select on validation and restore the best epoch, instead of using the last epoch?**
   Training longer can overfit; the last epoch is not necessarily the best model. Validation is
   held out from training, so it estimates generalization; picking the epoch and the
   configuration on it, and then evaluating once on test, keeps the test number an unbiased
   estimate of the chosen model's performance.
2. **What goes wrong if two conditions share one training function but different seeds?** Head
   initialization, shuffle order, and augmentation draws would all differ, and a small
   difference in validation accuracy could be luck rather than the factor under test. The
   harness passes the same `seed=SEED` everywhere so the factor is the only difference.
