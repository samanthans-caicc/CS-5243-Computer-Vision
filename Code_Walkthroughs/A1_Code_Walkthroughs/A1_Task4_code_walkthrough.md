# A1 Task 4 — Code Walkthrough

_Study note (not part of the graded submission). Explains `to_float01`, `to_uint8_safe`,
and the Task 4 notebook cell (`39aeb486`)._

---

## Key background: two different failures that both end in garbage pixels

Task 3 showed one way arithmetic goes wrong (`uint8` addition wraps). Task 4's job is to
show that this is only **half** the problem, and that the other half is a different
mechanism with a different fix. The analysis prompt asks you to keep them apart, so keep
them apart from the start:

| | mechanism | when it fires | symptom |
|---|---|---|---|
| **Integer overflow** | `uint8` arithmetic is modulo 256 | during the arithmetic, before you can clip | bright pixels become dark ones — `250 + 80 → 74` |
| **Unsafe float→uint8 cast** | `.astype(uint8)` truncates *and* wraps | at the conversion, on values outside `[0, 1]` | over-range pixels wrap to arbitrary values |

They are not the same bug wearing different clothes. Overflow happens because the *type*
had no headroom; the cast failure happens because the *value* left the range the conversion
assumes. Fixing one does not fix the other: computing in float removes the overflow but
still lets `1.4` become `102` at the cast, and clipping at the cast does nothing if you
already wrapped in `uint8` beforehand.

The pair `to_float01` / `to_uint8_safe` is the boundary between the two representations —
an integer container with a dtype-defined ceiling, and a normalized float where 0.0 is
black and 1.0 is white regardless of bit depth.

---

## `to_float01`

```python
arr = np.asarray(image)
if arr.dtype not in (np.uint8, np.uint16):
    raise ValueError(f"expected a uint8 or uint16 image, got {arr.dtype}")
# divide by the dtype maximum, not by the observed maximum: the scale is a property
# of the representation, so the same pixel maps to the same float in every image
return (arr.astype(np.float32) / np.float32(np.iinfo(arr.dtype).max)).astype(np.float32)
```

- **The dtype gate.** Only `uint8` and `uint16` have an unambiguous "full scale" to divide
  by. Handed a float array, the function would divide already-normalized data by 255 and
  silently return near-black; handed `int16`, the negative half of the range makes the
  whole idea meaningless. Same reasoning as Task 3's `_check_uint8`.
- **`np.iinfo(arr.dtype).max`** — 255 for `uint8`, 65535 for `uint16`. This is the single
  most important line in the function.

  Dividing by `arr.max()` instead would look identical on most images and be **wrong**:
  it would stretch every image to full range, so `color_shapes_dim.png` (max 121) and
  `color_shapes.png` (max 245) would come out equally bright and no longer comparable. The
  scale has to come from the representation, not from the content — the same argument as
  the `vmin`/`vmax` choice in the Task 1 figure.
- **`.astype(np.float32)` before dividing** so the division happens in float, not in
  integer space (`np.uint8(200) / 255` would be fine here since NumPy promotes, but making
  it explicit costs nothing and documents the intent).
- **The outer `.astype(np.float32)`** pins the return dtype. `np.float32(255)` as a divisor
  already keeps the result `float32`, but the public test asserts
  `to_float01(...).dtype == np.float32` and the cast makes that a guarantee rather than a
  consequence of NumPy's promotion rules.
- Result range is exactly `[0.0, 1.0]`: 0 maps to 0.0, and the dtype max maps to 1.0.

---

## `to_uint8_safe`

```python
arr = np.asarray(image)
if arr.dtype.kind != "f":
    raise ValueError(f"expected a floating-point image, got {arr.dtype}")
# clip BEFORE scaling so out-of-range values saturate instead of wrapping,
# and round in float64 so k/255 -> k round-trips exactly
scaled = np.clip(arr.astype(np.float64), 0.0, 1.0) * 255.0
return np.rint(scaled).astype(np.uint8)
```

- **`arr.dtype.kind != "f"`** accepts `float16/32/64` in one check. `.kind` is a
  single-character code (`"f"` float, `"u"` unsigned int, `"i"` signed int, `"b"` bool), so
  this is the idiomatic way to ask "is this a float of any width?" without listing three
  dtypes.
- **Clip first, then scale.** Clipping to `[0, 1]` *before* multiplying is equivalent to
  clipping to `[0, 255]` after, but it says what the function believes: the input is
  supposed to be a normalized image, and anything outside that is saturated rather than
  wrapped. It also means the multiplication can never produce a value the `uint8` cast
  cannot hold, so the cast is a pure type change.
- **`np.rint` before the cast.** `.astype(np.uint8)` *truncates* toward zero, so
  `0.9 * 255 = 229.5` would become `229` where rounding gives `230`, and a value that
  lands at `254.9999` because of float error would become `254` instead of `255`. Rounding
  explicitly is what makes the round trip exact. (Same half-to-even rounding as Task 3 —
  see that walkthrough for why ties go to even.)
- **Why `float64`.** This is the subtle one. `to_float01` returns `float32`, and
  `127/255` is not exactly representable in binary: in `float32` it is
  `0.49803921580314636`, which times 255 gives `126.99999…` in `float32` arithmetic.
  `np.rint` would still return 127 here, but the margin is thin and it is not thin for
  every value. Promoting to `float64` before the multiply gives ~16 significant digits
  instead of ~7, so every `k/255` lands close enough to `k` that rounding is unambiguous.
  That is what makes the public test's assertion — `to_uint8_safe(to_float01(x)) == x` for
  all 256 values — a guarantee rather than luck.

**Verified beyond the public test:** the round trip is exact for every value 0–255, and the
notebook reports `max absolute round-trip error = 0` on a real image.

---

## `record_metrics` — the cumulative-CSV helper

Defined in the Task 4 cell because Task 4 is the first writer; Experiment 2 calls the same
function later.

```python
def record_metrics(experiment, rows):
    fresh = pd.DataFrame(rows, columns=["experiment", "method", "condition", "metric", "value", "units"])
    if metrics_path.is_file():
        kept = pd.read_csv(metrics_path)
        kept = kept[kept["experiment"] != experiment]
        fresh = pd.concat([kept, fresh], ignore_index=True)
    fresh.to_csv(metrics_path, index=False)
    return fresh
```

- The assignment wants **one cumulative** `experiment_metrics.csv` that Task 4 creates and
  Experiment 2 appends to. The naive way to do that — `to_csv(..., mode="a")` — breaks the
  moment you re-run a cell: you get duplicate rows, and a reader cannot tell which run
  produced which number.
- This version is **idempotent**: it drops any existing rows whose `experiment` matches the
  one being written, then appends the fresh ones. Re-run Task 4 alone, re-run Experiment 2
  alone, or Restart-and-Run-All — the file always ends up with exactly one copy of each
  experiment's rows.
- The column list is fixed because `A1/config/A1.yml` declares
  `experiment_metrics.csv: [experiment, method, condition, metric, value, units]` and the
  validator errors on a missing column.
- The **long format** (one row per measurement, rather than one column per metric) is what
  lets two unrelated experiments share a file at all: Task 4's "levels" and Experiment 2's
  "fraction" live in the same `value`/`units` pair without either dictating the schema.

---

## The notebook cell logic (`39aeb486`)

Four comparisons on two images, in a deliberate order.

**1. The round trip that is safe by construction:**

```python
scene01 = sc.to_float01(scene)
restored = sc.to_uint8_safe(scene01)
round_trip_max_error = int(np.abs(restored.astype(np.int16) - scene.astype(np.int16)).max())
```

- `.astype(np.int16)` **before** subtracting is not optional. `uint8 - uint8` is still
  `uint8`, so `10 - 20` would give `246` instead of `-10` and the "max error" would be
  nonsense — the very bug the cell is about, sprung on the measurement code itself. Signed
  16-bit has room for the full `-255…255` range of possible differences.

**2. uint8 overflow vs. the same operation in float:**

```python
uint8_add = scene + np.uint8(OFFSET)                                  # wraps modulo 256
float_add = sc.to_uint8_safe(scene01 + np.float32(OFFSET / 255.0))    # clips at 1.0
wrapped = uint8_add < scene                                           # brightening made a pixel darker
```

- `wrapped = uint8_add < scene` is a neat detector: adding a positive offset can only make
  a value *larger* — unless it wrapped. Any sample where the "brightened" result is smaller
  than the original is direct evidence of overflow, and `.mean()` on the boolean array
  turns that into a fraction.
- The float path adds `OFFSET/255` in normalized space, which is the *same* brightening
  expressed in the other representation. That is what makes this a controlled comparison
  rather than two unrelated operations.

**3. The deliberately incorrect conversion the prompt names:**

```python
gained = (scene01 * np.float32(GAIN)).astype(np.float32)   # legitimate float image, range > 1
wrong = (gained * 255).astype(np.uint8)                    # no clip: truncates and wraps
right = sc.to_uint8_safe(gained)                           # clip -> round -> uint8
```

- `gained` is a perfectly reasonable float image — gain is how you brighten in normalized
  space — it simply is not confined to `[0, 1]` any more. That is the precondition
  `to_uint8_safe` enforces and the naive cast ignores.
- `(gained * 255).astype(np.uint8)` on a value of `1.4` computes `357.0`, and the C cast
  takes it modulo 256 → `101`. A pixel that should saturate to white becomes mid-grey. In
  the figure it shows up as the mottled dark-blue sky.

**4. The 16-bit ramp, where the scale is invisible in the values:**

```python
ramp_naive = ramp16.astype(np.uint8)        # keeps the low byte only
ramp_scaled = sc.to_uint8_safe(ramp01)      # correct 16 -> 8 bit rescale
ramp_monotonic = bool(np.all(np.diff(ramp_scaled[ramp_row].astype(np.int16)) >= 0))
```

- A ramp is the ideal probe because its correct answer is a *property*, not a picture:
  values must increase monotonically across the row. `np.diff(...) >= 0` tests exactly that,
  and `.astype(np.int16)` again avoids unsigned subtraction wrapping.
- `ramp16.astype(np.uint8)` keeps only the low byte of each 16-bit sample, so the smooth
  ramp becomes a repeating 0→255 sawtooth — 173 descents across one row. The second figure
  plots the row profile, which makes the sawtooth impossible to miss and impossible to
  mistake for "a bit darker."

---

## The measured result

From `outputs/tables/experiment_metrics.csv` (rows tagged `task4-dtype-range`):

| method | condition | metric | value |
|---|---|---|---|
| `to_float01` → `to_uint8_safe` | wide_scene.jpg | max absolute round-trip error | **0 levels** |
| uint8 addition | wide_scene.jpg +80 | fraction of samples wrapped darker | **0.5857** |
| float addition + `to_uint8_safe` | wide_scene.jpg +80 | max disagreement with uint8 addition | **255 levels** |
| `(float*255).astype(uint8)` | gain ×1.4 | fraction disagreeing with `to_uint8_safe` | **0.6289** |
| `(float*255).astype(uint8)` | gain ×1.4 | max disagreement | **255 levels** |
| `uint16 .astype(uint8)` | intensity_ramp_16bit.png | monotonic across middle row | **0 (false)** |
| `to_float01` → `to_uint8_safe` | intensity_ramp_16bit.png | monotonic across middle row | **1 (true)** |

Read that as three sentences: the safe pair round-trips exactly; **58.6%** of the image
wraps under a plain `uint8 + 80`; and **62.9%** of it is wrong under the unchecked cast,
with both failures reaching the full **255-level** maximum error. The ramp rows are the
control that shows the same rescaling logic is what makes a 16-bit image survive the trip
to 8 bits at all.

---

## What the analysis prompt is asking for

"Use observed values to explain overflow and conversion as two **distinct** failure
mechanisms." The trap is writing one paragraph that treats them as the same thing. The
numbers above separate them cleanly: 58.6% wrapping happens *in the arithmetic* and is
fixed by leaving `uint8` before adding; 62.9% disagreement happens *at the conversion* and
is fixed by clipping before the cast. Cite `dtype_range_experiment.png` for the visual and
the CSV rows for the values.

---

## Two concepts examiners love to probe

1. **Why divide by the dtype maximum, not the observed maximum** — because normalization
   must be a property of the representation. Dividing by `arr.max()` auto-stretches every
   image and destroys any comparison between a dim capture and a bright one.
2. **Why `.astype(np.uint8)` is not a conversion** — it truncates instead of rounding and
   it wraps instead of clipping, so it fails at both ends. The safe conversion is three
   operations, in order: clip, round, cast.
