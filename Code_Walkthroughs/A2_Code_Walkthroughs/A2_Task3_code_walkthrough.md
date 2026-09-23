# A2 Task 3 — Code Walkthrough

_Study note (not part of the graded submission). Explains `normalize_contrast`,
`histogram_equalize_gray`, and `apply_gamma` in `A2/src/student_code.py` and the Task 3
notebook cell (`d4b3d71b`)._

---

## Key background: three ways to move tones, and what each one costs

`exposure_under.png` has a luma mean of 0.15 — most of its values live in the bottom fifth
of the range. Three families of fix:

1. **Linear stretch** (`normalize_contrast`): pick a low and high value, map them to 0 and
   1, clip everything outside. Preserves the *shape* of the histogram, just widens it. Cost:
   whatever lies outside the chosen percentiles is **clipped** — lost for good.
2. **Gamma** (`apply_gamma`): `out = in^γ`. A monotone curve with no clipping at all;
   γ < 1 lifts shadows more than highlights. Cost: it compresses the top of the range, and
   it changes each channel's ratios, so colour saturation shifts.
3. **Histogram equalization** (`histogram_equalize_gray`): remap intensities so the output
   histogram is as flat as possible. Maximizes spread. Cost: it stretches whatever
   intensities are common — in a dark, noisy image that is the noise — and it can push a
   large fraction of pixels to the ceiling.

Two cross-cutting concerns the experiment has to address:

- **Noise amplification.** Any stretch multiplies the noise along with the signal. The
  notebook measures it as the std of the high-frequency residual (image minus a σ = 1 blur),
  relative to the original.
- **Luminance vs per-channel processing.** Equalizing R, G, B independently changes colour
  balance (each channel gets a different curve). Equalizing only luma and scaling the
  channels by the same gain keeps hue. The notebook does both and measures chromaticity
  shift.

---

## `normalize_contrast`

```python
if not (0.0 <= low_percentile < high_percentile <= 100.0): raise ValueError(...)
img = _check_image(image)
lo, hi = np.percentile(img.astype(np.float64), [low_percentile, high_percentile])
if hi == lo: raise ValueError(...)
out = (img.astype(np.float64) - lo) / (hi - lo)
return np.clip(out, 0.0, 1.0).astype(np.float32)
```

- `np.percentile` over the **whole array** — all pixels, all channels together. One `lo`,
  one `hi`, one affine map for every channel; that is the README's "jointly over every
  image value and channel", and it is what keeps colour ratios intact (a per-channel stretch
  would be a colour cast).
- The chained comparison encodes `0 ≤ low < high ≤ 100` in one expression.
- `hi == lo` (a flat image, or two percentiles that land on the same quantized value) would
  divide by zero — the contract says raise, so it raises.
- Clip after the map: values below `lo` become 0, above `hi` become 1. The *fraction* that
  gets clipped is the price of the stretch, and the notebook records it.

## `histogram_equalize_gray`

```python
a = np.asarray(image)
if a.ndim != 2 or a.dtype != np.uint8: raise ValueError(...)
hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
cdf = np.cumsum(hist)
cdf_min = cdf[np.nonzero(cdf)[0][0]]
total = cdf[-1]
if total == cdf_min:
    return a.copy()
lut = np.rint((cdf - cdf_min) / (total - cdf_min) * 255.0)
lut = np.clip(lut, 0, 255).astype(np.uint8)
return lut[a]
```

- `np.bincount(..., minlength=256)` is the fastest exact 256-bin histogram of a `uint8`
  array (no binning decisions to get wrong).
- The CDF is the running count. The classic formula
  `T(v) = round((cdf(v) − cdf_min) / (N − cdf_min) · 255)` subtracts the **first nonzero**
  CDF value so the darkest occupied level maps to exactly 0 and the brightest to exactly
  255 — the full range is used.
- `total == cdf_min` means only one grey level exists; the formula would be 0/0. The
  contract says return the image unchanged, so it does (a copy, never the caller's array).
- `np.rint` is round-half-to-**even** — the same deterministic rounding rule as A1, and the
  one the README names. Python's `round()` matches; `int(x + 0.5)` does not.
- `lut[a]` — build the 256-entry lookup table once, then apply it to every pixel with a
  single fancy-index. No per-pixel loop.

## `apply_gamma`

```python
if not (gamma > 0): raise ValueError(...)
img = np.clip(_check_image(image), 0.0, 1.0)
return np.power(img, np.float32(gamma)).astype(np.float32)
```

- Clip first so a stray 1.0001 does not produce something outside `[0, 1]`. On a
  normalized input `x^γ` stays in `[0, 1]` for any positive γ, so no clip is needed after.
- γ = 1 is the identity; γ < 1 brightens (every value moves toward 1); γ > 1 darkens. The
  public test checks all three.

---

## The notebook cell logic (`d4b3d71b`)

**Helpers.** `luminance()` is Rec.601 luma (`0.299R + 0.587G + 0.114B`, the same weights
`cv2.COLOR_RGB2GRAY` uses). `equalize_luminance()` equalizes the luma with the student
function, then multiplies every channel by `eq / lum` — one gain per pixel, so hue is
preserved. `equalize_per_channel()` runs `histogram_equalize_gray` on R, G, B separately.
`clahe_luminance()` is the **library comparison only**: `cv2.createCLAHE` on the L channel
of Lab, converted back.

**Nine results:** the original, three percentile stretches (0/100, 1/99, 2/98), two gammas
(0.5, 1/3), the two equalizations, and CLAHE. For each, `tonal_stats()` records:

- `clipped_low_fraction`, `clipped_high_fraction` — pixels at exactly 0 / 1 in the output.
- `luma_mean`, `luma_std`, `luma_p1/p50/p99`, `luma_entropy` (bits; flatter histogram →
  higher entropy, 8 is the max for 256 bins).
- `hf_residual_std` and `noise_amplification_ratio` (residual std relative to the original).
- `chromaticity_shift` — mean absolute change of `channel / (R+G+B)`.

Plus `input_fraction_outside_bounds` for the percentile stretches — how much of the *input*
the chosen percentiles throw away.

**The figure.** Top row: the original and seven results as images. Bottom row: each one's
64-bin luma histogram, titled with its clipped fractions.

---

## The measured result

| method | luma mean | clip lo / hi | entropy (bits) | noise ampl. | chroma shift |
|---|---|---|---|---|---|
| original | 0.149 | 0 / 0 | 4.99 | 1.00 | 0 |
| normalize 0–100 | 0.353 | 0.002 / 0.002 | 6.21 | 2.90 | 0.037 |
| normalize 1–99 | 0.342 | 0.057 / 0.028 | 6.27 | 3.09 | 0.059 |
| gamma 0.5 | 0.370 | 0 / 0 | 5.35 | 1.25 | 0.070 |
| gamma 1/3 | 0.511 | 0 / 0 | 5.17 | 1.14 | 0.096 |
| histeq (luma) | 0.490 | 0.000 / **0.163** | 5.62 | **5.98** | **0.012** |
| histeq (per channel) | 0.496 | 0.048 / 0.028 | **6.44** | 3.35 | 0.063 |
| CLAHE (L, clip 2, 8×8) | 0.178 | 0 / 0 | 5.59 | 1.33 | 0.026 |

What the table says:

- **Clipping vs brightness.** The 1–99 stretch throws away only 0.9 % of the *input* range
  but clips 8.5 % of *output* pixels — because the values it maps to 0 and 1 are exactly the
  crowded ends of a dark histogram. Gamma clips nothing and still lifts the mean to 0.37
  (γ = 0.5) or 0.51 (γ = 1/3).
- **Noise amplification tracks slope.** A linear stretch multiplies the residual by ~3×
  (the stretch factor). Luma equalization multiplies it by **6×** — its curve is steepest
  exactly where the dark pixels are. Gamma at 1.14–1.25× is the gentlest because `x^γ`
  flattens toward the top of the range where the noise is.
- **Luminance vs per-channel.** The luma-only equalization has the *lowest* chroma shift
  (0.012 — hue preserved by design) but pushes 16 % of pixels to the ceiling. The
  per-channel version has the highest entropy (6.44 bits, the flattest histogram) but a
  5× larger chroma shift — it re-balanced the colours.
- **CLAHE barely moves the mean** (0.15 → 0.18): it is a *local* method, redistributing
  contrast within tiles, not a global brightening. That is the right comparison to make
  and the wrong tool for this particular problem.
- **1–99 and 2–98 give identical numbers.** The image is 8-bit and dark: the 1st and 2nd
  percentiles land on the same quantized value, as do the 98th and 99th. A reminder that
  percentiles of a quantized image are step functions.

---

## Two concepts examiners love to probe

1. **Why subtract `cdf_min` in histogram equalization?** Without it the darkest occupied
   level maps to `cdf(v_min)/N · 255 > 0`, and the output never reaches black. Subtracting
   it (and renormalizing by `N − cdf_min`) pins the occupied range to the full `[0, 255]`.
2. **Global stretch, gamma, or equalization — which amplifies noise most, and why?** The
   one with the steepest tone curve where the pixels are. Equalization's curve *is* the
   CDF, and the CDF is steepest at the most populated intensities — in a dark image, the
   noisy shadows. The measured ratios (3.1× / 1.2× / 6.0×) are the evidence.
