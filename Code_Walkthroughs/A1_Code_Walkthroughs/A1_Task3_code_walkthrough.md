# A1 Task 3 — Code Walkthrough

_Study note (not part of the graded submission). Explains `brighten_loop`,
`brighten_vectorized`, and the Task 3 timing cell._

---

## Key background: uint8 arithmetic wraps

Task 2's big idea was views vs. copies. Task 3's is **what happens when a number leaves the
range its dtype can hold**. A `uint8` holds 0–255. Add past 255 and NumPy does not clamp —
it wraps around, silently:

```python
np.array([250], dtype=np.uint8) + np.uint8(20)   # → [14], because 270 mod 256 = 14
```

A bright pixel becomes a nearly black one. That's why *neither* implementation adds the
offset in `uint8`. Both move the value into floating point first, do the arithmetic there
where there's plenty of headroom, and only convert back at the very end — after rounding
and clipping have guaranteed the result actually fits.

The required order is: **add → round (`np.rint`) → clip to `[0, 255]` → cast to `uint8`.**

Why rounding must come before the cast: `.astype(np.uint8)` *truncates* toward zero, it
doesn't round. `30.7` would become `30`. So you round explicitly first, then the cast is
just a type change with no value change.

(Round-before-clip vs. clip-before-round happens to give the same answer here, because the
clip bounds 0 and 255 are whole numbers that rounding leaves alone. The spec names the
order, so follow it — but if asked, the honest answer is that these two commute for
integer bounds. The order that genuinely matters is that **both** come before the cast.)

---

## `_check_uint8` — the shared guard

```python
def _check_uint8(image):
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got {arr.dtype}")
    return arr
```

- Same pattern as `_check_rectangle` in Task 2: an underscore-prefixed internal helper so
  both brighten functions validate identically instead of duplicating the check.
- It also does the `np.asarray` conversion and hands the array back, so each caller starts
  with one line instead of two.
- Why guard at all: the clip range `[0, 255]` and the final `uint8` cast only make sense
  for 8-bit input. Handed the `uint16` ramp (0–65535), the function would silently crush
  almost everything to 255. Failing loudly is better than quietly producing garbage.

---

## `brighten_loop`

```python
arr = _check_uint8(image)
shift = float(offset)
source = arr.reshape(-1)                      # one flat loop covers 2-D and 3-D alike
result = np.empty(source.size, dtype=np.uint8)
for i in range(source.size):
    value = float(np.rint(float(source[i]) + shift))   # round half to even, then clip
    if value < 0.0:
        value = 0.0
    elif value > 255.0:
        value = 255.0
    result[i] = value
return result.reshape(arr.shape)
```

- `arr.reshape(-1)` — flatten to 1-D. The `-1` means "you figure out the length." This is
  the trick that lets **one** loop handle a grayscale `(H, W)` image and an RGB
  `(H, W, 3)` image without separate nested-loop versions. Every element gets the same
  treatment regardless of which axis it lives on, so the shape doesn't matter until the end.
- `np.empty(...)` allocates the output without initializing it — slightly cheaper than
  `np.zeros`, and safe here because the loop writes every single element.
- `float(source[i])` — this is the important line. `source[i]` is a `np.uint8` scalar;
  converting to a Python `float` moves the arithmetic out of 8-bit space so
  `250 + 20` gives `270.0`, not `14`.
- `np.rint(...)` rounds half-to-even (see below), then the outer `float(...)` unwraps the
  `np.float64` it returns back to a plain Python float.
- The `if/elif` pair is the clip, written out longhand — this is the loop version, so
  clipping is done by hand rather than with `np.clip`.
- `result[i] = value` — assigning a whole-number float into a `uint8` slot. Safe because
  the clip already guaranteed `0 ≤ value ≤ 255` and the rounding already made it integral.
- `result.reshape(arr.shape)` — restore the original shape at the end.
- Note the input is never written to: everything goes into `result`. Non-mutating, same as
  Task 2's requirement.

---

## `brighten_vectorized`

```python
arr = _check_uint8(image)
shifted = arr.astype(np.float64) + float(offset)
return np.clip(np.rint(shifted), 0.0, 255.0).astype(np.uint8)
```

Exactly the same four steps, with each one applied to the whole array at once:

| step | loop version | vectorized version |
|---|---|---|
| escape uint8 | `float(source[i])` | `arr.astype(np.float64)` |
| add offset | `+ shift` | `+ float(offset)` |
| round | `np.rint(...)` per element | `np.rint(shifted)` on the array |
| clip | `if/elif` per element | `np.clip(..., 0.0, 255.0)` |
| back to uint8 | `result[i] = value` | `.astype(np.uint8)` |

- `arr.astype(np.float64)` makes a float **copy** — the original `uint8` array is untouched.
- `+ float(offset)` **broadcasts**: the single scalar is applied to every element without a
  Python loop (same broadcasting idea as `replace_region` in Task 2).
- `np.rint` and `np.clip` are *ufuncs* — they loop over the array in compiled C, not in
  Python.

### Why `float64` in both

This is deliberate, and it's the reason the two functions produce byte-identical output.
The loop version uses Python `float`, which **is** a C double = `np.float64`. If the
vectorized version used `float32` instead, the two could round differently on exact `.5`
boundaries where float32's coarser precision nudges a value to one side. Matching the
precision makes "the outputs are identical" a guarantee, not a coincidence — and the whole
timing comparison depends on both paths computing the same answer.

---

## `np.rint` and round-half-to-even

`np.rint` rounds to the nearest integer, and on an exact tie it rounds to the nearest
**even** integer:

```python
[0, 1, 2, 3, 4] + 0.5  →  [0.5, 1.5, 2.5, 3.5, 4.5]  →  rint  →  [0, 2, 2, 4, 4]
```

So `0.5→0`, `1.5→2`, `2.5→2`, `3.5→4`. Not what school taught ("always round .5 up"), but
it's the IEEE-754 default and it's what the spec asks for.

**Why half-to-even:** always rounding `.5` up biases results upward. Over a whole image
with a fractional offset, half-up would systematically brighten the mean; half-even sends
ties up and down about equally, so the error cancels out. Unbiased, not arbitrary.

---

## The notebook cell logic

**The timing harness:**

```python
def time_call(function, image, offset, repeats):
    samples, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(image, offset)
        samples.append((time.perf_counter() - start) * 1000.0)
    return result, samples
```

- `time.perf_counter()` is the right clock for this: high resolution and *monotonic* (it
  never jumps backwards the way a wall-clock can). Difference × 1000 = milliseconds.
- Returns both the final result and every sample, so the cell can check correctness *and*
  report timing from the same run.

**What makes the comparison controlled** — this is what the analysis prompt is asking about:

- **Same image** (`wide_scene.jpg`, loaded once into `source`), same `OFFSET`, same
  `REPEATS` for both. Only the implementation strategy varies.
- Both functions apply the identical rounding-and-clipping rule, so neither is winning by
  cutting a corner.
- `assert np.array_equal(loop_out, vector_out)` runs **before** any timing is reported. A
  speed comparison between two functions that compute different things would be
  meaningless; this makes the comparison legitimate.

**Median, not mean:**

```python
loop_median = float(np.median(loop_ms))
```

One repeat can be hit by a garbage-collection pause or the OS scheduling something else —
you can see this in the CSV's `max_ms` column, which is noticeably higher than `min_ms`.
The mean would be dragged by that outlier; the median ignores it. Reporting `min`, `median`,
and `max` together also lets a reader see how noisy the measurement was.

**The CSV** (`outputs/tables/timing.csv`) — one row per implementation, carrying both the
measurement and the conditions it was measured under: image, shape, element count, offset,
repeat count, median/min/max ms, `outputs_identical`, and `speedup_vs_loop`. Recording the
conditions alongside the numbers is what makes the row reproducible rather than a bare
stopwatch reading.

**The probe at the end** prints `[[0, 10, 250]]` brightened by 20 through both
implementations (`[[20, 30, 255]]` — the 250 saturates) next to a naive `uint8` add
(`[[20, 30, 14]]` — the 250 wraps). That's the representation point the timing is in
service of: the dtype rule is what makes the output *correct*, and both implementations
enforce it identically.

---

## The measured result

Roughly **211 ms (loop) vs. 2.9 ms (vectorized) — about 70× — on 648,000 elements, with
identical outputs.** Exact numbers shift run to run; the CSV is the record.

**Why the gap is that big.** Per element, the loop makes Python do real work: index the
array with bounds checking, box the raw byte into a `np.uint8` object, convert to a Python
`float`, dispatch a full `np.rint` ufunc call *on a single scalar*, run two comparisons,
then cast and store. Six-hundred-thousand times, all through the interpreter. The
vectorized version dispatches each operation **once** and then runs a tight compiled loop
over contiguous memory, with no per-element Python object anywhere. The gap is interpreter
overhead, not better arithmetic — the arithmetic is identical, which is precisely what the
equality assertion proves.

Worth stating plainly in the analysis: this is a *representation* argument, not a
benchmarking contest (the spec says so outright). The point is that choosing to operate on
the array as a whole, rather than element by element, is what makes image-sized work
practical at all.

---

## Two concepts examiners love to probe

1. **Integer overflow / wraparound** — be ready to say what `uint8(250) + 20` gives (`14`,
   because arithmetic is mod 256) and why the fix is to compute in float and clip, not to
   hope the values stay small. Also why `.astype(np.uint8)` alone is not a fix: it
   truncates rather than rounds, and it doesn't clip.
2. **Why vectorized code is faster** — the answer is *per-element Python interpreter
   overhead* (boxing, dispatch, bounds checks), not "NumPy uses a better algorithm." Both
   do the same 648,000 additions; one does them in C, the other pays Python's per-element
   tax on every single one.
