# A1 Experiment 2 — Code Walkthrough

_Study note (not part of the graded submission). Explains `hsv_rule` and the Experiment 2
notebook cell (`c5372f53`)._

---

## Key background: OpenCV's HSV, and why hue is the odd one out

HSV re-expresses a color as **hue** (which color), **saturation** (how pure), and **value**
(how bright). The appeal for a selection rule is that "red" is one hue interval, whereas in
RGB it is a three-dimensional cone of combinations.

Two facts about OpenCV's 8-bit HSV, both of which the code has to respect:

1. **The ranges are 0–179 for hue and 0–255 for saturation and value.** Hue is halved so a
   full 360° circle fits in a `uint8`. A "hue of 170" is 340° — nearly red, not the
   yellow-green you might guess if you assumed 0–255.
2. **Hue is circular; saturation and value are not.** Red sits at the seam: it occupies
   roughly 170–179 *and* 0–10. Any interval that contains red therefore has a **lower bound
   greater than its upper bound**, and a naive `lower <= h <= upper` test selects nothing at
   all. That wrap case is the part of this function being graded.

The experiment then asks the harder question: a rule is tuned on one image, so what is it
actually a property of — the object, or the capture?

---

## `hsv_rule`

```python
arr = _check_rgb(hsv_image)
low = np.asarray(lower, dtype=np.int32)
high = np.asarray(upper, dtype=np.int32)
if low.shape != (3,) or high.shape != (3,):
    raise ValueError(f"lower and upper must each hold 3 values, got {low.shape} and {high.shape}")
hue, sat, val = (arr[:, :, i].astype(np.int32) for i in range(3))
```

- `_check_rgb` is reused: an HSV image has the same `(H, W, 3)` layout as an RGB one. The
  name is about the *shape*, not the color space.
- **`int32` for both the bounds and the channels.** Comparing `uint8` pixels against a
  Python `int` works, but as soon as a bound is passed as a NumPy `uint8` (or someone later
  adds a tolerance) unsigned arithmetic can wrap and turn a comparison into nonsense.
  Promoting both sides to signed 32-bit makes every comparison unambiguous, at negligible
  cost.
- The generator unpacking `hue, sat, val = (arr[:, :, i]... for i in range(3))` splits the
  three channel planes in one line; each is `(H, W)`.

```python
# saturation and value are ordinary inclusive intervals
mask = (sat >= low[1]) & (sat <= high[1]) & (val >= low[2]) & (val <= high[2])
if low[0] <= high[0]:
    hue_mask = (hue >= low[0]) & (hue <= high[0])
else:
    # OpenCV hue is 0-179 and circular: lower > upper means the interval crosses 0
    # (for example 170 -> 10 is "red"), so accept either side of the wrap point
    hue_mask = (hue >= low[0]) | (hue <= high[0])
return mask & hue_mask
```

- **`&` and `|`, not `and` / `or`.** Python's keywords call `bool()` on their operands,
  which raises `ValueError: truth value of an array is ambiguous` for arrays. The bitwise
  operators are the element-wise versions, and each comparison produces a full boolean
  array, so the whole rule is evaluated for every pixel at once — no loop.
- **Inclusive on both ends** (`>=` and `<=`), as the spec asks. Worth stating because
  `cv2.inRange` is also inclusive; a rule written with `<` would quietly select fewer
  pixels than the same numbers passed to OpenCV.
- **The wrap branch is the whole trick.** For an interval like 170 → 10, the set of accepted
  hues is 170–179 **union** 0–10, so the two comparisons are joined with `|` (union) rather
  than `&` (intersection). Written the ordinary way, `h >= 170 and h <= 10` is unsatisfiable
  and the mask is empty everywhere.
- Saturation and value never wrap, so they keep the plain `&` form in both branches. Only
  hue gets the special case, because only hue is an angle.
- The return is a plain boolean `(H, W)` array — directly usable as `image[mask]`, as
  `mask.mean()` for a fraction, or as an `imshow` panel.

**Verified:** on a probe row of hues `[0, 5, 90, 175, 179]`, the rule `(170, 50, 50) →
(10, 255, 255)` selects hues 0, 5 and 175 and rejects 90; the non-wrapped rule
`(80, …) → (100, …)` selects only 90.

---

## The notebook cell logic (`c5372f53`)

**One rule, three captures:**

```python
RULE_LOWER, RULE_UPPER = (170, 100, 60), (10, 255, 255)   # wrapped hue: 170..179 and 0..10
CONDITIONS = {"normal": "color_shapes.png", "dim": "color_shapes_dim.png", "warm": "color_shapes_warm.png"}
```

The rule is chosen the way anyone would choose it: measured on the **normal** image, where
the red target sits at hue 0, saturation 204, value 225, and then given comfortable margins
(S ≥ 100, V ≥ 60). It is then applied *unchanged* to all three images. That is the
controlled part — the rule is the constant, the capture condition is the only variable, and
the three images are pixel-aligned so the same ROI means the same object in each.

**Conversion, with the convention handled:**

```python
hsv = cv2.cvtColor(sc.bgr_to_rgb(rgb), cv2.COLOR_BGR2HSV)   # cv2 expects BGR input
```

Same point as Experiment 1: imageio hands back RGB, `COLOR_BGR2HSV` promises BGR input, so
the channels are reversed first. Skipping this would compute hue from a red/blue-swapped
image — the target would land near hue 120 (blue) and the rule would select nothing, for
entirely the wrong reason.

**Three metrics, because "selected fraction" alone cannot fail informatively:**

```python
selected_fraction = float(mask.mean())                       # how much of the image is selected
hit = int((mask & target_mask).sum())
coverage = hit / int(target_mask.sum())                      # how much of the target it keeps
precision = hit / int(mask.sum()) if mask.any() else 0.0     # how much of what it keeps is target
```

- `mask.mean()` on a boolean array is the fraction of `True` — no need to sum and divide.
- `target_mask` is the ROI from `color_regions.yml` (`target_roi_xyxy = [55, 65, 225, 245]`),
  built once as a boolean rectangle. Because the three images are aligned, one mask serves
  all three.
- **Coverage** (recall) and **precision** answer different failure modes. A rule can fail by
  losing the target (coverage falls) or by grabbing things that are not the target
  (precision falls) — and the warm and dim conditions fail in those two opposite ways. A
  single number could not have shown that.
- The `if mask.any()` guard exists because the dim condition selects **zero** pixels, and
  `hit / 0` would raise.

**Clause-level attribution — the part that turns a result into a diagnosis:**

```python
hue_ok = float(((roi[:, 0] >= RULE_LOWER[0]) | (roi[:, 0] <= RULE_UPPER[0])).mean())
sat_ok = float(((roi[:, 1] >= RULE_LOWER[1]) & (roi[:, 1] <= RULE_UPPER[1])).mean())
val_ok = float(((roi[:, 2] >= RULE_LOWER[2]) & (roi[:, 2] <= RULE_UPPER[2])).mean())
```

Each clause of the rule is re-tested **on its own**, restricted to pixels inside the target
ROI. When the dim condition returns an empty mask, these three numbers say *why*: hue still
passes on 100% of target pixels and value on 92.2%, but saturation passes on **0%**. The
failure is attributable to one clause, not to "HSV being unreliable" — which is the
difference between an observation and an explanation.

**Recording and plotting:**

- `record_metrics("exp2-hsv-stability", rows)` — the idempotent helper defined in the Task 4
  cell. It replaces this experiment's rows in `experiment_metrics.csv` and leaves Task 4's
  rows alone, so the cumulative file stays correct no matter what order cells are run in.
- The figure is 3 columns (conditions) × 3 rows: the image with the shared ROI drawn on it,
  the resulting mask, and a histogram of the target ROI's saturation and value with the
  rule's lower bounds drawn as vertical lines. That third row is the evidence for the
  attribution above — you can *see* the dim image's saturation cluster sitting just left of
  the S ≥ 100 line.

---

## The measured result

| condition | selected fraction | ROI coverage | precision | S clause in ROI | V clause in ROI |
|---|---|---|---|---|---|
| normal | 0.1102 | **92.2%** | **100%** | 92.2% | 92.2% |
| dim | 0.0000 | **0%** | 0% | **0%** | 92.2% |
| warm | 0.1381 | **100%** | **86.6%** | 100% | 100% |

Three genuinely different outcomes from one rule:

- **normal** — what the rule was tuned for. It keeps 92.2% of the target and selects
  nothing else (precision 1.0). The missing 7.8% is not error: those 2,401 ROI pixels are
  the rectangle's solid black outline stroke, every one of them at value exactly 30 and
  saturation under 100. They are inside the ROI rectangle but they are not red, and a rule
  for "red" should reject them. Note this is why the S and V clause pass rates for the
  normal condition are both 92.2% — the same border pixels fail both.
- **dim** — total collapse. The dimming multiplied the scene by 0.56 *and* desaturated it,
  so the target's saturation falls from 204 to 85, under the rule's lower bound of 100. The
  value clause still passes, so this is not "too dark to see" — it is "too grey for a
  saturation threshold tuned on a vivid capture."
- **warm** — the opposite failure. Coverage rises to 100% (the warm cast pushes the target
  *deeper* into the red hue window), but precision drops to 86.6%: the same cast drags the
  dark outlines of the other shapes into the hue interval too. The rule now selects the
  target **and** ~4,750 pixels that are not the target.

The one-line summary: the rule is not a property of the red rectangle. It is a property of
the rectangle *under the lighting it was tuned on*, and it degrades in two different
directions depending on which way the capture moves.

---

## What the analysis prompt is asking for

"Explain the strongest and weakest condition." The strongest is **normal** on precision
(1.000) or **warm** on coverage (1.000) — say which and why they are not the same claim.
The weakest is unambiguously **dim**, and the clause pass rates let you name the mechanism
in one sentence: saturation, not hue and not value. Cite `hsv_stability.png` and the
`exp2-hsv-stability` rows of `experiment_metrics.csv`.

---

## Two concepts examiners love to probe

1. **Hue wraparound.** Why a red interval has `lower > upper`, why the test becomes a union
   (`|`) instead of an intersection (`&`), and what a naive `low <= h <= high` returns for
   red (an empty mask). Also that OpenCV hue is 0–179, not 0–255 or 0–360.
2. **Why a threshold rule is a statement about the capture, not the object.** The same
   three numbers select 92% of the target under one light and 0% under another. Any claim
   that a color rule "detects red" needs the capture condition attached to it.
