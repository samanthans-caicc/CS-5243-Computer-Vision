# A3 Experiment 8 — Code Walkthrough

_Study note (not part of the graded submission). Explains `HFImageDataset`, `_hf_image_column`,
and `build_external_dataloaders` in `src/student_code.py`, the provided
`a3_tools.resplit_by_lesion`, and the Experiment 8 notebook cell (`a1e5f002`). This is a
controlled classroom exercise, not a diagnostic tool._

---

## Key background: why this dataset, and why lesion-level splits

`marmal88/skin_cancer` is derived from HAM10000: 10,015 dermatoscopic images in 7 diagnostic
classes. Two properties make it a hard test of the same pipeline:

- **Class imbalance.** Most images are benign moles (*melanocytic nevi*). A model can score a
  high accuracy by mostly predicting the majority class, while failing the rare classes. This
  is where Task 4's per-class metrics and macro-F1 stop being formalities.
- **Several images per lesion.** Many lesions were photographed more than once. If one photo of
  a lesion lands in training and another in test, the test score partly measures *recognizing
  that lesion again*. This is **data leakage**. `resplit_by_lesion` groups images by
  `lesion_id` so each lesion appears in exactly one split.

This is also the one approved network call in the assignment: only `build_external_dataloaders`
may download, and only the two named datasets. The model weights still come from the local
files.

---

## `a3_tools.resplit_by_lesion` (provided)

```python
pooled = datasets.concatenate_datasets([raw[name] for name in ("train", "validation", "test")])
rows = pooled.select_columns(["image_id", "lesion_id", "dx"]).to_pandas()
rows = rows.drop_duplicates("image_id").sort_values("image_id")
folds = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SKIN_CANCER_SPLIT_SEED)
...  fold 0 -> test, fold 1 -> validation, folds 2-6 -> train
```

- The published splits are **pooled** and re-cut, because they are image-level.
- `StratifiedGroupKFold` makes 7 folds where every **group** (`lesion_id`) stays inside one
  fold, while trying to keep each fold's class proportions close to the overall ones
  (**stratified**). One fold is test, one is validation, five are training: about 71 / 14 / 14 %,
  giving **7153 / 1431 / 1431** images.
- Only three small columns are converted to pandas; the images are never decoded during the
  split. The final `pooled.select(...)` is index-based.
- A fixed course seed means every student gets identical splits.

---

## `build_external_dataloaders`

```python
if augmentation not in AUGMENTATION_LEVELS: raise ValueError(...)
import datasets
raw = datasets.load_dataset(hf_dataset_name)
if hf_dataset_name == "marmal88/skin_cancer":
    raw = a3_tools.resplit_by_lesion(raw)
```

- Augmentation is validated **before** the download, so a bad argument fails in milliseconds
  instead of after a 3.7 GB fetch.
- `datasets` is imported inside the function. Everything else in `student_code.py` works
  without it installed, and the network dependency is visibly confined to this function.
- `load_dataset` downloads on first use and reads from the Hugging Face cache afterwards — the
  cache lives outside `A3/` (see the README).

```python
if "validation" not in raw or "test" not in raw:
    pieces = {"train": raw["train"], ...existing splits...}
    needed = [name for name in ("validation", "test") if name not in pieces]
    parts = pieces["train"].train_test_split(test_size=0.15 * len(needed), seed=int(seed), shuffle=True)
    ...  # one missing split -> the 15 % piece; two missing -> 30 %, halved into validation and test
```

- The docstring requires carving any missing split **deterministically from train with `seed`**.
  Neither A3 dataset needs it (both arrive with three splits), but the branch makes the function
  correct for any Hugging Face dataset with only a `train` split.

```python
feature = raw["train"].features[label_column]
if isinstance(feature, datasets.ClassLabel):
    class_names, label_to_index = list(feature.names), None
else:
    observed = set()
    for split in raw.values():
        observed.update(split.unique(label_column))
    class_names = [str(value) for value in sorted(observed)]
    label_to_index = {value: index for index, value in enumerate(sorted(observed))}
```

The two datasets store labels differently, and the rest of the pipeline needs integers
`0 … K−1`:

- **EuroSAT** uses a `ClassLabel` feature: labels are already integers, and the feature carries
  the ordered names. No mapping is needed.
- **Skin Cancer's `dx`** is a plain **string** column. The code collects the distinct values from
  **every** split (so a class that happened to be absent from train would still get an index),
  sorts them for a stable order, and builds `string → index`. Sorting is what makes the label
  numbering identical on every run and every machine.

```python
image_column = _hf_image_column(raw["train"])
transform_of = {"train": training_transform(augmentation), "validation": evaluation_transform(),
                "test": evaluation_transform()}
for split in ("train", "validation", "test"):
    dataset = HFImageDataset(raw[split], image_column, label_column, class_names,
                             transform_of[split], label_to_index)
    loaders[split] = DataLoader(..., shuffle=split == "train",
                                generator=_loader_generator(seed) if shuffle else None, ...)
```

- **The same contract as `build_dataloaders`**, built from the same helpers: augmentation only
  on train, deterministic ImageNet-normalized evaluation, seeded shuffling, the one-sample-batch
  guard. That shared contract is what lets `run_training` treat these loaders like the bundled ones.
- `_hf_image_column` finds the column whose feature type is `datasets.Image` (falling back to
  common names), so the code does not hard-code `"image"` for each dataset.

### `HFImageDataset`

```python
def __getitem__(self, index):
    row = self.split[int(index)]
    image = row[self.image_column]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")
    raw_label = row[self.label_column]
    label = self.label_to_index[raw_label] if self.label_to_index is not None else int(raw_label)
    return self.transform(image), int(label)
```

- Rows are read **lazily**: one image is decoded from the cache when requested, so the whole
  dataset never has to fit in memory. Decoding is CPU work every epoch, which is why the
  notebook uses 6 workers here.
- `int(index)` — some samplers pass NumPy integers, which Hugging Face indexing rejects.
- `.convert("RGB")` normalizes grayscale or RGBA images to three channels.
- `class_names` is kept on the dataset because the function only returns loaders; the notebook
  reads `skin_loaders["train"].dataset.class_names`.

---

## The notebook cell logic (`a1e5f002`)

```python
EXTERNAL_EPOCHS, EXTERNAL_WORKERS = 3, 6
skin_loaders = sc.build_external_dataloaders("marmal88/skin_cancer", "dx", BATCH_SIZE,
                                             BEST_AUGMENTATION, SEED, EXTERNAL_WORKERS)
skin_train_counts = external_label_counts(skin_loaders["train"])
SKIN_RUN = run_training("exp8_skin_cancer", architecture=BEST_ARCHITECTURE, mode="finetune",
                        augmentation=BEST_AUGMENTATION, optimizer_name=BEST_OPTIMIZER,
                        epochs=EXTERNAL_EPOCHS, seed=SEED, loaders=skin_loaders,
                        num_classes=len(SKIN_CLASSES), class_names=SKIN_CLASSES)
skin_evaluation, skin_metrics = test_report(SKIN_RUN)
```

- **`mode="finetune"` is set explicitly**, not taken from `BEST_MODE` (frozen): the task says to
  fine-tune. Architecture, augmentation, and optimizer are the selected ones; SGD fine-tuning
  uses the table's `lr = 1e-3`. Three epochs bound the GPU time.
- The new head has 7 outputs: trainable parameters are 4,016,515
  (= 4,007,548 backbone + 1280 · 7 + 7).
- **The kept checkpoint is still chosen by validation accuracy** — the same `run_training`
  rule as everywhere else. On imbalanced data, accuracy rewards majority-class performance, so
  this selection criterion does not look at rare-class performance at all.
- The figure sorts classes by training count and plots per-class test F1 on a second y-axis,
  with macro-F1 as a dashed line, so count and F1 can be read against each other directly.
- Records: 10-metric run block + 5 summary rows + 7 classes × 3 = 36 Experiment 8 rows.

---

## What it produced

```
training images per class {'actinic_keratoses': 233, 'basal_cell_carcinoma': 368,
  'benign_keratosis-like_lesions': 785, 'dermatofibroma': 81, 'melanocytic_Nevi': 4790,
  'melanoma': 795, 'vascular_lesions': 101}     imbalance ratio 59.1x
test accuracy 0.7952 | macro F1 0.4792 | majority-class share of training data 0.670
```

| class | train images | test support | precision | recall | F1 |
|---|---|---|---|---|---|
| melanocytic_Nevi | 4790 | 957 | 0.870 | **0.968** | **0.916** |
| melanoma | 795 | 159 | 0.637 | 0.365 | 0.464 |
| benign_keratosis-like | 785 | 157 | 0.592 | 0.535 | 0.562 |
| basal_cell_carcinoma | 368 | 73 | 0.485 | 0.644 | 0.553 |
| actinic_keratoses | 233 | 47 | 0.630 | 0.362 | 0.460 |
| vascular_lesions | 101 | 21 | 0.667 | 0.286 | 0.400 |
| dermatofibroma | 81 | 17 | **0.000** | **0.000** | **0.000** |

- A model that always answered "melanocytic_Nevi" would score 957 / 1431 = **0.669** on this
  test split. The fine-tuned model's 0.795 is 0.126 above that.
- The majority class has F1 0.92; every other class is between 0.40 and 0.56, and dermatofibroma
  is 0 — none of its 17 test images was recognized. Macro-F1 averages those seven values
  equally, giving 0.479.
- Best validation accuracy was at epoch 3, the last epoch of the budget.

---

## Two concepts examiners love to probe

1. **What would an image-level split have done to the test score?** Images of the same lesion
   would appear in both train and test, so some "test" images would be near-duplicates of
   training images. The score would rise, but it would no longer measure performance on *new
   lesions*. Grouped splitting is the standard defence whenever samples are not independent.
2. **Why does the same model look reasonable by accuracy and poor by macro-F1?** Accuracy weights
   each class by its support, so the 957-image majority class dominates it. Macro-F1 weights each
   class equally, so the six minority classes — and one class at 0 — dominate it. Neither number
   is wrong; they answer different questions.
