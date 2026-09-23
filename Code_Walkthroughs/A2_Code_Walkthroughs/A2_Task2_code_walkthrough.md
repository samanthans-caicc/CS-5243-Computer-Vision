# A2 Task 2 — Code Walkthrough

_Study note (not part of the graded submission). Explains `add_gaussian_noise`,
`add_impulse_noise`, and `median_filter_manual` in `A2/src/student_code.py` and the Task 2
notebook cell (`060e1aeb`)._

---

## Key background: two noise models, and why one filter cannot serve both

- **Additive Gaussian noise** perturbs *every* pixel by a small, zero-mean amount:
  `y = clip(x + N(0, σ²))`. Averaging neighbours cancels it (the mean of *n* independent
  samples has variance σ²/n), which is why a Gaussian blur helps — but the blur also
  averages across edges, so it trades noise for detail.
- **Impulse (salt-and-pepper) noise** leaves most pixels untouched and replaces a fraction
  *p* with 0 or 1. Averaging is exactly the wrong tool: one 0 in a 3×3 window drags the mean
  down by a third of the pixel value. A **median** ignores outliers entirely as long as
  fewer than half the window is corrupted, so it removes impulses without smearing them.
- **Bilateral filtering** (`cv2.bilateralFilter`) is a Gaussian blur whose weights are also
  reduced for neighbours with a different *intensity* (`sigmaColor`). It smooths flat
  regions while preserving edges — but an impulse is itself a large intensity difference,
  so the filter *protects* it instead of removing it.

The reason the rankings differ by noise type is not a detail; it is the point of the task.

The other background idea is **reproducibility**: every random value comes from the
`np.random.Generator` the caller passes in. Same seed → same noisy image, bit for bit, on
any machine. The notebook declares its seeds (5243 and 5244) so a grader can reproduce the
observations.

---

## `add_gaussian_noise`

```python
if not (sigma >= 0): raise ValueError(...)
img = _check_image(image)
noise = rng.normal(0.0, float(sigma), size=img.shape).astype(np.float32)
return np.clip(img + noise, 0.0, 1.0).astype(np.float32)
```

- One `rng.normal` draw with the image's full shape — for a colour image each channel gets
  independent noise, which is the standard sensor model.
- `sigma = 0` is legal and returns the input (clipped). `NaN` sigma fails the
  `not (sigma >= 0)` test.
- Clipping is part of the contract: a normalized image cannot hold −0.05 or 1.1.

## `add_impulse_noise`

```python
if not (0.0 <= probability <= 1.0): raise ValueError(...)
img = _check_image(image)
H, W = img.shape[:2]
selected = rng.random((H, W)) < probability      # which pixels
salt = rng.random((H, W)) < 0.5                  # 1 or 0
values = np.where(salt, np.float32(1.0), np.float32(0.0))
out = img.copy()
if img.ndim == 3:
    out[selected] = values[selected][:, None]    # broadcast across channels
else:
    out[selected] = values[selected]
return out.astype(np.float32)
```

- Two spatial draws, both `(H, W)` — **not** `(H, W, C)`. The selection is per *pixel*, and
  a selected colour pixel receives the same value on every channel (the README's "all
  channels receive the same impulse"; the public test checks the three channels are equal).
- `rng.random() < probability` selects each pixel independently with exactly that
  probability; `< 0.5` makes salt and pepper equiprobable. With `probability = 1.0` every
  pixel is selected (`random() < 1.0` is always true), which is the test's
  `{0.0, 1.0}` check.
- `values[selected]` is a 1-D vector of the selected pixels' new values; `[:, None]` gives
  it a trailing axis so it broadcasts over the `(n, C)` slice that `out[selected]` addresses
  on a 3-D image.
- `img.copy()` first — the caller's array is untouched.

## `median_filter_manual`

```python
k = _check_odd_size(kernel_size, "kernel_size")
img = _check_image(image)
r = k // 2
padded = _pad_spatial(img, r, border_mode)
H, W = img.shape[:2]
stack = np.empty((k * k,) + img.shape, dtype=np.float32)
idx = 0
for i in range(k):
    for j in range(k):
        stack[idx] = padded[i:i + H, j:j + W]
        idx += 1
return np.median(stack, axis=0).astype(np.float32)
```

- Same slice-walking pattern as `convolve2d_manual`, but instead of weighting and summing
  the `k²` shifted slices, they are **stacked** along a new leading axis. `stack[:, y, x]`
  is then exactly the `k×k` neighbourhood of pixel `(y, x)`, flattened.
- `np.median(axis=0)` sorts each pixel's neighbourhood and takes the middle. It is a
  reduction, not a filter — the windowing, padding, and border logic are all student code,
  which is what "do not delegate filtering" asks for.
- Memory: `k² · H · W · 4` bytes — for `k = 7` on the 255×383 test pattern that is ~19 MB.
  Fine here; a reason to prefer a histogram-based median for large images.
- Colour images work with no extra code: the channel axis rides along in `img.shape`, and
  the median is taken per channel.

---

## The notebook cell logic (`060e1aeb`)

**Observations.** Own draws from the two noise functions with declared seeds and the
prescribed parameters (σ = 0.08, p = 0.07) on `test_pattern_gray.png`; the distributed
`gaussian_noisy.png` / `impulse_noisy.png` are loaded too, but only for the figure — the
README says they are independent draws, so they are never used for metrics.

**Metrics.** `quality()` returns MSE, RMSE, PSNR, and SSIM (`data_range=1.0` — the
scikit-image functions need to be told the image is in `[0, 1]`, or PSNR is silently wrong).

**Three denoisers, each swept:**

```python
sweep = {
    "gaussian":  σ ∈ {0.5, 1.0, 1.5, 2.5}           (own convolution, support 2⌈3σ⌉+1)
    "median":    k ∈ {3, 5, 7}                        (own median)
    "bilateral": (sigmaColor, sigmaSpace) ∈ {(.05,3), (.1,3), (.2,3), (.1,6)}, d = 7
}
```

The `lambda im, s=s:` default-argument idiom freezes each parameter value into its closure
— without it every lambda would see the *last* value of the loop variable.

`cv2.bilateralFilter` is called on the `float32` image, so `sigmaColor` is in intensity
units of `[0, 1]` (0.1 = "treat neighbours more than ~0.1 apart as a different surface").

Every setting on both noise types is written to the CSV (four quality metrics plus
`edge_energy_ratio_vs_clean` as the detail-preservation measure). The **figure** shows, per
noise type, the clean image, the distributed observation, the own observation, and each
method at its **best-PSNR setting for that noise type** — chosen by `max(..., key=psnr)`
so the figure never favours a method through a hand-picked parameter.

---

## The measured result

**Gaussian noise (σ = 0.08), input PSNR 22.1 dB / SSIM 0.43:**

| method | best setting | PSNR | SSIM |
|---|---|---|---|
| bilateral | sigmaColor 0.2, sigmaSpace 3 | **32.7** | **0.84** |
| median | k = 3 | 26.4 | 0.67 |
| Gaussian | σ = 0.5 | 24.2 | 0.52 |
| Gaussian | σ = 1.0 | 20.0 | 0.73 |

**Impulse noise (p = 0.07), input PSNR 16.1 dB / SSIM 0.38:**

| method | best setting | PSNR | SSIM |
|---|---|---|---|
| median | k = 3 | **25.2** | **0.98** |
| Gaussian | σ = 0.5 | 19.2 | 0.43 |
| bilateral | sigmaColor 0.2 | 16.9 | 0.44 |

Things worth being able to point at:

- **The ranking flips.** Bilateral wins by 6 dB on Gaussian noise and is nearly useless on
  impulse noise (+0.8 dB over the noisy input): its intensity weight treats every impulse as
  an edge to protect. Median wins on impulse noise with SSIM 0.98 — nearly perfect
  structure — because a single outlier per window never becomes the median.
- **Gaussian blur at σ = 1 scores *below* the noisy input on PSNR** (20.0 vs 22.1) while
  its SSIM goes *up* (0.43 → 0.73). This test pattern is mostly one-pixel stripes and a
  checkerboard; the blur removes noise *and* the structure, and PSNR punishes the second
  more than SSIM does. The two metrics disagree, and that disagreement is the
  detail-preservation trade-off in numbers.
- **Median k = 7 destroys the image** (12.6 dB on Gaussian, 11.3 dB on impulse — both far
  below the noisy input). Once the window is wider than the stripes, the stripes *are* the
  outliers. Task 7 uses this as its second controlled failure.

---

## Two concepts examiners love to probe

1. **Why is the median robust to impulses and the mean not?** The mean is a linear
   combination — one extreme value moves it proportionally. The median is an order
   statistic — an extreme value only shifts which element sits in the middle by one
   position, and with fewer than `k²/2` outliers the middle element is still a clean pixel.
2. **Why does the same generator produce the same image, and why does the assignment
   care?** `np.random.default_rng(seed)` is a deterministic PCG64 stream; the two `random()`
   calls consume it in a fixed order. "All randomness from the supplied generator" is what
   makes the experiment reproducible by a grader — and it is also why the distributed noisy
   files (a different stream) are for eyes only.
