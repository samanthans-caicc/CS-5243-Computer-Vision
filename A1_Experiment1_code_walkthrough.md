# A1 Experiment 1 — Code Walkthrough

_Study note (not part of the graded submission). Explains `_check_rgb`, `grayscale_mean`,
`grayscale_luminance`, and the Experiment 1 notebook cell (`be37e794`)._

---

## Key background: two things that both "change the colors", for unrelated reasons

Experiment 1 puts five renderings of one image side by side. Two different ideas are being
separated, and the analysis prompt asks which evidence tells them apart:

- **Channel order is a convention, not data.** The bytes `(235, 35, 35)` are red under RGB
  and blue under BGR. Nothing about the array says which; the *loader* and the *display*
  have to agree. imageio gives RGB, OpenCV expects and returns BGR. Getting this wrong
  swaps red and blue and leaves green untouched — a large, obvious, *reversible* error.
- **Grayscale weighting is a modelling choice.** Collapsing three channels to one requires
  deciding what "brightness" means. An unweighted mean treats the channels as equally
  important; BT.601 luminance weights them 0.299/0.587/0.114 because human vision is far
  more sensitive to green than to blue. Both are "correct"; they answer different
  questions, and the difference between them is small and *irreversible* (you cannot get
  the colors back either way).

One is a bug. The other is a decision. The experiment is designed so you can point at
numbers that distinguish them.

---

## `_check_rgb` — the shared guard

```python
def _check_rgb(image):
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {arr.shape}")
    return arr
```

- Third helper of this shape in the module (`_check_rectangle` in Task 2, `_check_uint8` in
  Task 3). Same reasoning: three functions need the identical check, so it lives in one
  place and hands the converted array back.
- Rejecting a 2-D array matters here — a grayscale image passed to `grayscale_mean` would
  otherwise average across the *width* axis and return a column of numbers rather than an
  error.
- Note it deliberately does **not** check dtype. `grayscale_mean` and `grayscale_luminance`
  are used on both `uint8` arrays and `float32` `[0, 1]` arrays in this notebook, and both
  are legitimate.

---

## `grayscale_mean`

```python
arr = _check_rgb(image_rgb)
# float32 accumulation, so the caller's value scale (0-255 or 0-1) is preserved
return arr.astype(np.float32).mean(axis=2, dtype=np.float32)
```

- `axis=2` averages **across channels**, collapsing `(H, W, 3)` to `(H, W)`. Getting the
  axis wrong is the classic slip: `axis=0` would average the rows and return a `(W, 3)`
  array — still an array, no exception, complete nonsense.
- `dtype=np.float32` tells `mean` to accumulate in float32 rather than promoting to
  float64. The cast on the way in already guarantees float arithmetic (so a `uint8` input
  cannot overflow while summing three values), and the explicit accumulate dtype pins the
  return type without a second cast.
- **Scale-preserving by design.** The function does not normalize. Given `uint8` input it
  returns values in `[0, 255]`; given `[0, 1]` floats it returns `[0, 1]`. That is what
  lets the notebook control the display scale in one place — it converts once with
  `to_float01` and every downstream panel is automatically on the same scale.

---

## `grayscale_luminance`

```python
arr = _check_rgb(image_rgb).astype(np.float32)
weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)   # ITU-R BT.601 luma
return (arr @ weights).astype(np.float32)
```

- **`arr @ weights` is the whole computation.** The `@` operator is matrix multiplication;
  with an `(H, W, 3)` array and a `(3,)` vector, NumPy contracts the last axis of the left
  operand against the vector and returns `(H, W)`. It is exactly
  `0.299*R + 0.587*G + 0.114*B` computed for every pixel at once, in compiled code — the
  same vectorization argument as Task 3, applied to a weighted sum instead of an addition.
- The weights sum to 1.000, which is why the output stays in the input's range: a white
  pixel `(1, 1, 1)` maps to 1.0, not to 3.0.
- **Where the numbers come from:** ITU-R BT.601, the standard behind the luma channel of
  analog and early digital video. Green dominates because the human eye's cones are most
  sensitive in the green band; blue contributes least. This is why a pure blue region looks
  much darker than a pure green one in the luminance panel but identical in the mean panel.
- OpenCV's `COLOR_BGR2GRAY` uses the same BT.601 weights, which is what makes the
  measured agreement below meaningful.

---

## The notebook cell logic (`be37e794`)

**One source, one scale:**

```python
source_rgb = iio.imread(paths["color_shapes.png"])   # uint8 RGB as stored on disk
rgb01 = sc.to_float01(source_rgb)                    # float32 [0, 1] display scale
```

Converting once at the top is what makes "hold the display limits fixed" enforceable: every
panel below is a float in `[0, 1]`, so every `imshow` can use `vmin=0.0, vmax=1.0` and the
comparison is honest. Mixing `uint8` panels (0–255) with float panels (0–1) would force
different limits per panel, and matplotlib's autoscaling would quietly rescale each one.

**The five representations:**

```python
correct = rgb01                                                    # (a)
bgr_as_rgb = sc.to_float01(sc.bgr_to_rgb(source_rgb))              # (b)
mean_gray = sc.grayscale_mean(rgb01)                               # (c)
luma_gray = sc.grayscale_luminance(rgb01)                          # (d)
opencv_gray = cv2.cvtColor(sc.bgr_to_rgb(source_rgb), cv2.COLOR_BGR2GRAY)   # (e)
opencv01 = sc.to_float01(opencv_gray)
```

- **(b)** reuses Task 2's `bgr_to_rgb` to *simulate* the mistake: reverse the channels, then
  display the result as if it were RGB. That is precisely what happens when an OpenCV-loaded
  image is handed straight to `matplotlib`.
- **(e) is the one people get wrong.** `cv2.COLOR_BGR2GRAY` promises "my input is BGR." Our
  array is RGB, so passing it directly would ask OpenCV to treat red as blue — the
  weights would land on the wrong channels. `sc.bgr_to_rgb(source_rgb)` reverses the
  channels to produce a genuinely BGR array, which is what the conversion is documented to
  receive. (The function name reads oddly here; the operation is its own inverse — reversing
  three channels is a swap of the outer two either way.)
- `opencv_gray` comes back `uint8`, so `to_float01` puts it on the shared `[0, 1]` scale
  before it is displayed or compared.

**The control that makes the convention error measurable:**

```python
opencv_gray_wrong = cv2.cvtColor(source_rgb, cv2.COLOR_BGR2GRAY)   # wrong convention
opencv_convention_gap = np.abs(opencv_gray.astype(np.int16) - opencv_gray_wrong.astype(np.int16))
```

This is the sharpest evidence in the experiment. Two grayscale images, both plausible, both
looking like reasonable grayscale renderings of the same scene — and they differ on
**97.2%** of pixels by up to **39 levels**. A channel-order error survives the trip to
grayscale as a *brightness* error, where nothing looks obviously swapped and only a
measurement finds it. (`.astype(np.int16)` before subtracting, for the same unsigned-wrap
reason as Task 4.)

**The comparison table** collects seven statistics into a DataFrame — mean and max
difference for the channel-order pair, mean and max for the two grayscale rules, agreement
between OpenCV and our luminance, and the two convention-gap numbers. Mean *and* max
together are what let a reader tell "small everywhere" from "small on average, large
somewhere," which is exactly the difference between the grayscale rules and the
channel-order error.

**The sixth panel is a diagnostic, not a representation:**

```python
upper = float(data.max()) if position == 5 else 1.0
ax.imshow(data, cmap=cmap, vmin=0.0, vmax=upper)
```

The `|luminance − mean|` difference image peaks at 0.146, so on the shared `[0, 1]` scale
it renders as a near-black rectangle. It is stretched to its own maximum and *labelled as
stretched*, because the rule "hold the display limits fixed" applies to the five things
being compared — not to a diagnostic that exists to show where they differ. Saying so in
the panel title is what keeps that honest.

---

## The measured result

| comparison | statistic | value |
|---|---|---|
| channel order: RGB vs BGR-as-RGB | mean \|difference\| | **0.1413** (float01) |
| channel order: RGB vs BGR-as-RGB | max \|difference\| | **0.8235** (= 210/255) |
| grayscale rule: luminance − mean | mean signed difference | **+0.0041** |
| grayscale rule: luminance − mean | max \|difference\| | **0.1458** |
| OpenCV gray (BGR in) vs our luminance | max \|difference\| | **0.0019** (≈ 0.5 level) |
| OpenCV gray: right vs wrong convention | max \|difference\| | **39 levels** |
| OpenCV gray: right vs wrong convention | fraction of pixels differing | **0.9723** |

The separation the analysis prompt is after falls straight out of that table:

- The **channel-order error is an order of magnitude larger** than the grayscale-rule
  difference (max 0.82 vs 0.15) and it is a swap, not a shift — red and blue exchange
  places while green sits still.
- The **two grayscale rules differ by at most 0.146** and by only +0.004 on average. They
  disagree most on saturated blue and green (visible in the diagnostic panel), which is
  exactly where the BT.601 weights depart hardest from 1/3 each.
- **Our luminance and OpenCV's grayscale agree to 0.0019**, about half a `uint8` level —
  i.e. they implement the same formula and differ only by OpenCV's rounding to integers.
  That agreement is what licenses the claim that the weights are standard rather than
  invented, and it is the reference against which the 39-level convention gap is damning.

---

## Two concepts examiners love to probe

1. **Channel order is a convention that lives outside the array.** Be ready to say what
   imageio returns, what OpenCV expects, what the array itself knows (nothing), and what the
   error looks like: red/blue swapped in color, and a *silent* brightness error once the
   image is converted to grayscale.
2. **Why 0.299/0.587/0.114 and not 1/3 each.** Human sensitivity peaks in green; luminance
   models perceived brightness, the mean models arithmetic average. Neither is a bug — but
   the two answer different questions, and the max difference of 0.146 is where they part
   company.
