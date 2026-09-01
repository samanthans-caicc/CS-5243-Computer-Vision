# A1 Task 1 — Code Walkthrough

_Study note (not part of the graded submission). Explains `image_summary` in
`A1/src/student_code.py` and the Task 1 notebook cell (`0456cb0b`)._

---

## Key background: an image is an array with three separate facts attached

Every later task depends on being able to say, precisely, what an image array *is*. Three
facts, and they are independent of one another:

1. **Shape** — NumPy reports `(height, width)` or `(height, width, channels)`. Rows first.
   Almost every non-NumPy tool (screen resolutions, image editors, ROI metadata, OpenCV's
   `Size`) writes `width × height`. `portrait_scene.png` is `(540, 360, 3)`: 540 rows of
   360 pixels — a *tall* image, even though "540, 360" looks wide if you read it as W×H.
2. **Dtype** — how many bits each sample gets, and therefore the range the representation
   can hold: `uint8` → 0–255, `uint16` → 0–65535. This is a property of the *container*,
   not of the picture.
3. **Observed range** — the min and max actually present. `wide_scene.jpg` happens to span
   the full 0–255; `color_shapes_dim.png` only reaches 121. A dim image is not a different
   dtype, it is the same container holding smaller numbers.

Task 1 is the task that forces you to report all three separately instead of conflating
them. Everything downstream (Task 4's dtype safety, Experiment 2's saturation thresholds)
is a consequence of keeping them apart.

---

## `image_summary`

```python
arr = np.asarray(image)
if arr.ndim == 2:                       # grayscale: no channel axis
    height, width = arr.shape
    channels = 1
elif arr.ndim == 3:                     # color: (H, W, C) with C == 3 or 4
    height, width, channels = arr.shape
else:
    raise ValueError(f"Malformed image array. Expected a 2-D or 3-D image array, got shape {arr.shape!r}")
```

- `np.asarray` — cheap guarantee the input behaves like an ndarray (no copy if it already
  is one). Same opening move as every other function in the module.
- The `ndim` branch is the whole reason this function exists: a grayscale image is 2-D and
  has **no** channel axis, so `height, width, channels = arr.shape` would raise
  `ValueError: not enough values to unpack`. Reporting `channels = 1` for 2-D is a
  *convention* the function commits to, and it is what lets `intensity_ramp_16bit.png` fit
  the same CSV row shape as the RGB images.
- The `else` raises rather than guessing. A 1-D or 4-D array is not an image; failing
  loudly beats emitting a summary that quietly describes nothing.

```python
return {
    "shape": tuple(int(n) for n in arr.shape),
    "height": int(height),
    "width": int(width),
    "channels": int(channels),
    "dtype": str(arr.dtype),
    "range": (arr.min().item(), arr.max().item()),
    "mean": float(arr.mean()),
    "nbytes": int(arr.nbytes),
}
```

- **`int(...)` and `.item()` everywhere.** `arr.shape` already holds Python ints, but
  `arr.min()` returns a *NumPy scalar* (`np.uint8(30)`), not a Python `int`. `.item()`
  unwraps it. This matters the moment the dict goes into a DataFrame → CSV → JSON: a NumPy
  scalar can serialize awkwardly or fail outright, and it drags its dtype around with it.
  Converting at the boundary keeps the summary a plain-Python object.
- **`str(arr.dtype)`** — `"uint8"`, not `dtype('uint8')`. Again: a string is what belongs
  in a table.
- **`arr.mean()` is `float64` regardless of input dtype.** NumPy accumulates the mean in
  double precision, so a `uint8` image cannot overflow while being averaged. `float(...)`
  just unwraps it.
- **`nbytes`** is `size × itemsize` — the memory the array occupies, *not* the file size on
  disk. `wide_scene.jpg` is 648,000 bytes in RAM and a fraction of that on disk, because
  JPEG is compressed. Worth being able to state that distinction out loud.
- The dict returns `range` as a tuple; the notebook cell unpacks it into two separate CSV
  columns (`minimum`, `maximum`), because the config's schema asks for scalar columns.

---

## The notebook cell logic (`0456cb0b`)

**Loading:**

```python
image_paths = a1_tools.discover_images()
for path in image_paths:
    array = iio.imread(path)
```

- `discover_images()` returns the asset paths sorted, so the CSV row order is deterministic
  — the same run twice produces the same file, which is what "reproducible artifact" means.
- `iio.imread` (imageio v3) **preserves the native dtype** and returns RGB. That is why the
  16-bit ramp survives as `uint16` instead of being silently downshifted to 8 bits, and why
  no `bgr_to_rgb` call is needed here (BGR is OpenCV's convention, not imageio's — see
  Experiment 1).

**Building the table:**

```python
s = sc.image_summary(array)
low, high = s["range"]
records.append({"filename": path.name, "shape": s["shape"], ..., "mean": round(s["mean"], 4), ...})
metadata = pd.DataFrame.from_records(records)
metadata.to_csv(csv_path, index=False)
```

- One dict per image → `DataFrame.from_records` → CSV. The column names are not arbitrary:
  `A1/config/A1.yml` declares
  `image_metadata.csv: [filename, shape, height, width, channels, dtype, minimum, maximum, mean, bytes]`,
  and `validate_submission.py` checks the header. Extra columns are fine; a missing one is
  a validation **error**.
- `index=False` keeps the anonymous 0,1,2… index out of the file.
- `round(s["mean"], 4)` — four decimals is plenty for a mean pixel value, and it keeps the
  CSV readable.

**Three named pixels:**

```python
co = loaded["channel_order.png"]
named_pixels = {
    "channel_order.png RED block  (row 150, col 120)": co[150, 120],
    ...
}
print(f"{label}: array value (R, G, B) = {tuple(int(c) for c in value[:3])}")
```

- `co[150, 120]` indexes **row 150, column 120** — a length-3 vector, one pixel's RGB.
  Naming the coordinates "row/col" rather than "x/y" in the label is deliberate; Task 5's
  first failure is exactly what happens when someone mixes those up.
- The three samples come from the red, green, and blue blocks of `channel_order.png`, and
  they print `(235, 35, 35)`, `(35, 210, 65)`, `(40, 90, 240)`. That is the evidence that
  imageio really did hand back RGB: the "red" block is genuinely large in channel **0**.
  If the loader were returning BGR, the red block would read `(35, 35, 235)`.

**The figure:**

```python
if a.ndim == 2:  # grayscale / uint16: scale to [0, 1] for display only
    ax.imshow(a, cmap="gray", vmin=0, vmax=np.iinfo(a.dtype).max)
else:
    ax.imshow(a[..., :3])
```

- `np.iinfo(a.dtype).max` asks the **dtype** for its ceiling (65535 for `uint16`) instead
  of using the observed maximum. Displaying with `vmax=arr.max()` would auto-stretch each
  image and make a dim image look identical to a bright one — the exact confusion Task 1 is
  supposed to prevent. Same principle as `to_float01` in Task 4.
- `a[..., :3]` drops a fourth alpha channel if one exists, so `imshow` never gets a
  4-channel array it would interpret as RGBA.
- The `suptitle` states the caption the rubric asks for: shape order, dtype, and the fact
  that the grayscale panel is scaled *for display only*.

---

## The measured result

Seven images, from `outputs/tables/image_metadata.csv`:

| file | shape | dtype | min–max | mean |
|---|---|---|---|---|
| channel_order.png | (300, 720, 3) | uint8 | 35–255 | 86.47 |
| color_shapes.png | (400, 640, 3) | uint8 | 30–245 | 179.68 |
| color_shapes_dim.png | (400, 640, 3) | uint8 | 16–121 | 100.23 |
| color_shapes_warm.png | (400, 640, 3) | uint8 | 7–255 | 165.47 |
| intensity_ramp_16bit.png | (256, 512) | uint16 | 0–65535 | 23065.51 |
| portrait_scene.png | (540, 360, 3) | uint8 | 35–240 | 190.93 |
| wide_scene.jpg | (300, 720, 3) | uint8 | 0–255 | 162.16 |

Three things in that table are worth being able to point at:

- **`intensity_ramp_16bit.png` is the only 2-D row** (`channels = 1`) and the only `uint16`
  row. Its mean of 23,065 is meaningless next to the others' 100–190 *unless* you also
  quote the dtype — 23,065/65,535 ≈ 0.35, comparable to 90/255 ≈ 0.35. Same brightness,
  different container. This is the clearest argument for normalizing before comparing
  (Task 4).
- **`color_shapes_dim.png` tops out at 121**, less than half the range its `uint8` container
  allows. The dtype did not change; the *content* did. Experiment 2 is where that matters.
- **`portrait_scene.png` is (540, 360, 3)** — taller than it is wide. The one file in the
  set that punishes reading shape as width-first.

---

## Two concepts examiners love to probe

1. **`(height, width)` vs `(width, height)`** — be ready to say which one NumPy uses, which
   one the rest of the world uses, and what goes wrong when they are crossed: not a crash,
   but an in-bounds crop of the wrong region (demonstrated in Task 5).
2. **Dtype vs. observed range vs. displayed appearance** — three different things.
   `uint16` does not mean "bright"; a maximum of 121 does not mean "8 bits is too small";
   and an `imshow` that auto-scales hides both. Quote `intensity_ramp_16bit.png` and
   `color_shapes_dim.png` as the two examples.
