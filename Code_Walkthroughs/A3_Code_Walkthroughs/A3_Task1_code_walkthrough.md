# A3 Task 1 — Code Walkthrough

_Study note (not part of the graded submission). Explains `evaluation_transform`,
`training_transform`, `_loader_generator`, `_drop_last`, `_worker_init`, `ArrayImageDataset`,
and `build_dataloaders` in `src/student_code.py`, plus the Task 1 notebook cell (`aa4cd4ea`)._

---

## Key background: what "deterministic" has to cover

A data pipeline feeds three kinds of randomness into training, and each one has to be
pinned for two runs to be comparable:

1. **Which images go in which batch** — the training shuffle order.
2. **What each training image looks like** — random flips, crops, rotations, jitter.
3. **What evaluation sees** — this must have *no* randomness at all. Validation and test
   are measuring instruments; a measuring instrument that changes between readings is useless.

The contract is therefore asymmetric: training is random **but reproducible** (same seed →
same sequence), and evaluation is not random at all. Every experiment in Section 5 changes
exactly one factor; that only works if the pipeline does not quietly change a second one.

There is also a size question. The bundled images are **64×64**, but both pretrained
backbones were trained on ImageNet at **224×224**. Every split is resized to 224 so that
the network sees inputs at the scale its filters were learned at — and, just as
importantly, so that every split and every condition goes through the same resize.

---

## The transforms

```python
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

def _normalization():
    return [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

def evaluation_transform(image_size=IMAGE_SIZE):
    return transforms.Compose([transforms.Resize((image_size, image_size))] + _normalization())
```

- `ToTensor` converts a PIL `uint8` image `(H, W, 3)` in `[0, 255]` to a `float32` tensor
  `(3, H, W)` in `[0, 1]` — it **moves the channel axis first**, which is the layout every
  PyTorch convolution expects.
- `Normalize` then computes `(x − mean) / std` per channel using the **ImageNet** statistics.
  These are not this dataset's statistics, on purpose: the pretrained filters expect inputs
  distributed the way ImageNet inputs were. The notebook's printed range `(−2.001, 2.64)` is
  exactly what that produces — `(1 − 0.406)/0.225 = 2.64` is a saturated blue channel.
- `Resize((224, 224))` with a **tuple** forces both sides; `Resize(224)` with an int would
  resize only the shorter side. For square 64×64 inputs it makes no difference, but the tuple
  states the intent.
- `_normalization()` is shared so the two transforms cannot drift apart.

```python
def training_transform(augmentation, image_size=IMAGE_SIZE):
    if augmentation not in AUGMENTATION_LEVELS: raise ValueError(...)
    if augmentation == "none":
        return evaluation_transform(image_size)
    if augmentation == "moderate":
        steps = [Resize((s, s)), RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5),
                 ColorJitter(brightness=0.2, contrast=0.2)]
    else:   # strong
        steps = [RandomResizedCrop(s, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
                 RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5), RandomRotation(30),
                 ColorJitter(0.4, 0.4, 0.4, hue=0.1), RandomGrayscale(0.1)]
    return transforms.Compose(steps + _normalization())
```

- **`none` is literally the evaluation transform.** That makes the Exp 3 "none" condition
  identical to the evaluation pipeline, so any difference between levels is attributable to
  augmentation and nothing else.
- **`moderate` is label-preserving for every class.** A horizontal flip turns horizontal
  stripes into horizontal stripes, vertical into vertical, `/` diagonals into `\` diagonals
  (still "diagonal"), and leaves checkers, circles, and dots in their class. Brightness and
  contrast jitter change appearance, not geometry.
- **`strong` is deliberately aggressive.** `RandomResizedCrop` replaces the plain resize: it
  crops 60–100 % of the image area and resizes the crop to 224, so pattern frequency changes.
  `RandomRotation(30)` tilts stripes by up to 30°, which starts to blur the line between
  horizontal/vertical and diagonal. That is the point — it gives Exp 3 a level that can hurt.
- Validation happens **before** anything is built, so a typo like `"Strong"` fails
  immediately instead of after a ten-epoch run.

---

## Three small helpers that make the loaders reproducible

```python
def _loader_generator(seed):
    generator = torch.Generator(); generator.manual_seed(int(seed)); return generator

def _drop_last(size, batch_size, shuffle):
    return bool(shuffle and batch_size > 1 and size % batch_size == 1)

def _worker_init(worker_id):
    base = torch.initial_seed() % 2**32
    np.random.seed(base); random.seed(base)
```

- **`_loader_generator`** gives each training loader its **own** RNG. With a private
  generator, the shuffle order depends only on `seed` — not on how many random numbers some
  earlier cell happened to consume from the global RNG. The `DataLoader` also draws each
  epoch's worker base seed from this generator, so augmentation randomness inside workers is
  tied to the same seed.
- **`_drop_last`** drops the final batch only when it would contain **exactly one** sample.
  BatchNorm in training mode cannot compute a variance from one value and raises an error.
  For this dataset, 720 % 32 = 16, so nothing is dropped; the guard is there for
  `build_external_dataloaders`, where split sizes are arbitrary.
- **`_worker_init`** runs once inside each worker process. PyTorch already seeds `torch` in
  each worker (`initial_seed()` is the per-worker seed), but **not** NumPy or Python's
  `random`. Torchvision's transforms use `torch`'s RNG, so this is belt-and-braces: any
  NumPy-based transform added later would also be reproducible.

---

## `ArrayImageDataset`

```python
class ArrayImageDataset(Dataset):
    def __init__(self, images, labels, transform, class_names):
        self.images = np.ascontiguousarray(images)
        self.labels = np.asarray(labels).astype(np.int64)
        ...
    def __getitem__(self, index):
        image = Image.fromarray(self.images[index]).convert("RGB")
        return self.transform(image), int(self.labels[index])
```

- Torchvision transforms operate on **PIL images** (or tensors), so each array is wrapped
  with `Image.fromarray` on demand. `.convert("RGB")` is a no-op for 3-channel input but
  guarantees three channels if a grayscale array ever comes through.
- Labels are stored as `int64` because `nn.CrossEntropyLoss` requires `Long` targets.
- `class_names` lives on the dataset so any later code holding only a loader can recover the
  label names (`loader.dataset.class_names`) — the notebook's harness relies on this.
- The class is public (no underscore) because the notebook reuses it: Experiment 7's noisy
  test set and Task 6's corrupted-label and rotated loaders are all `ArrayImageDataset`s.

---

## `build_dataloaders`

```python
archive = np.load(Path(dataset_path))
transform_of = {"train": training_transform(augmentation),
                "validation": evaluation_transform(), "test": evaluation_transform()}
for split in ("train", "validation", "test"):
    labels = archive[f"{split}_labels"]
    dataset = ArrayImageDataset(archive[f"{split}_images"], labels, transform_of[split], names)
    shuffle = split == "train"
    loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, generator=_loader_generator(seed) if shuffle else None,
        worker_init_fn=_worker_init if num_workers else None,
        drop_last=_drop_last(len(dataset), batch_size, shuffle), ...)
```

- **One dictionary decides which split gets which transform.** Augmentation can only reach
  `train` because the other two keys are hard-wired to `evaluation_transform()`.
- **Only `train` shuffles.** Validation and test are read in file order, every time. That
  is what makes `predictions.csv` row `test_0000` the first test image in the archive, and
  what lets `extract_embeddings` return labels "in loader order" that match `evaluate_model`.
- Class names come from `a3_tools.dataset_metadata()`. The `try/except` fallback to
  `"0", "1", …` only matters if the metadata file is missing; it keeps the function usable on
  a bare NPZ.
- `pin_memory=False`, `persistent_workers=False`: the safe defaults. Workers are torn down at
  the end of each epoch, which costs a little time but means no state leaks between runs.

---

## The notebook cell logic (`aa4cd4ea`)

The cell is a set of **checks**, each one aimed at a sentence of the task contract:

| Printed line | Contract it checks |
|---|---|
| `split sizes {'train': 720, 'validation': 180, 'test': 180}` | three splits, correct sizes |
| `train batch (32, 3, 224, 224) float32 \| labels (32,) int64` | `(images, labels)` batches, channel-first, Long targets |
| `ImageNet-normalized range (−2.001, 2.64)` | normalization actually applied |
| the three `augmentation=…` lines + one `eval` line | augmentation changes **only** train; eval stays `Resize, ToTensor, Normalize` |
| `same seed -> identical … True \| different seed -> different order: True` | shuffling controlled by `seed` |
| `validation order deterministic: True` | evaluation is not shuffled |

The seed check builds a fresh loader each time and takes its **first batch of labels**.
Two loaders with seed 17 must produce identical label sequences; seed 99 must not. That is the
same property `test_dataloaders_are_reproducible` checks in the public tests (which compares
images too).

---

## Two concepts examiners love to probe

1. **Why ImageNet statistics instead of this dataset's own mean and std?** The backbone's
   first convolution was trained on inputs normalized with ImageNet statistics; its weights
   encode that input distribution. Re-normalizing with the synthetic dataset's statistics
   would shift every activation away from what the pretrained filters expect. Transfer
   learning reuses the *whole* input contract, not just the weights.
2. **Why is "random but reproducible" different from "deterministic"?** Training must still
   see shuffled batches and random augmentations — that variety is what training needs. Seeding
   only makes the random sequence *repeatable*, so two conditions differ in their intended
   factor and not in luck. Evaluation, by contrast, is fully deterministic: no shuffle, no
   random transforms, so a number measured on validation means the same thing every time.
