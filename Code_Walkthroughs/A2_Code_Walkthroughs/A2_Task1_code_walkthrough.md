# A2 Task 1 — Code Walkthrough

_Study note (not part of the graded submission). Explains `convolve2d_manual`,
`gaussian_kernel`, and `sharpen_image` in `A2/src/student_code.py` and the Task 1
notebook cell (`5bdce6fe`)._

---

## Key background: convolution is correlation with the kernel flipped

A linear filter slides a small kernel `k` over the image and takes a weighted sum at every
position. There are two ways to line the kernel up:

- **Correlation:** `out[y, x] = Σ k[i, j] · img[y + i − r, x + j − r]` — the kernel is laid
  on top of the image as-is.
- **Convolution:** `out[y, x] = Σ k[i, j] · img[y − i + r, x − j + r]` — the kernel is
  **rotated 180°** first (`k[::-1, ::-1]`), then correlated.

For a symmetric kernel (any Gaussian) the two are identical, which is why most people never
notice the difference. For an asymmetric kernel they are not, and the assignment demands
true convolution — so the flip has to be there, and the experiment has to *prove* it is
there. `cv2.filter2D` computes **correlation**, so the honest library reference for a
convolution is `filter2D` on the *flipped* kernel.

Two other ideas the task hinges on:

- **Border handling.** The kernel hangs off the edge of the image at the boundary, so the
  image must be padded first. `reflect` (NumPy) mirrors *without* repeating the edge pixel
  (`d c b | a b c d`), which is exactly OpenCV's `BORDER_REFLECT_101`. `edge` repeats the
  boundary value (`a a a | a b c d`); `constant` pads zeros. The mode you pass your own
  function and the `borderType` you pass OpenCV must match or the comparison is meaningless.
- **Unsharp masking.** `sharpened = img + amount · (img − blur)`. The term `img − blur` is
  the high-pass detail; adding it back scaled by `amount` steepens edges. It is linear in
  `amount`, and it is *not* clipped inside the function, so overshoot past `[0, 1]` — the
  halo — is preserved as evidence rather than hidden.

---

## `gaussian_kernel`

```python
size = _check_odd_size(size, "size")
if not (sigma > 0):
    raise ValueError(...)
r = size // 2
ax = np.arange(-r, r + 1, dtype=np.float64)
g1 = np.exp(-(ax ** 2) / (2.0 * float(sigma) ** 2))
k = np.outer(g1, g1)
k /= k.sum()
return k.astype(np.float32)
```

- `_check_odd_size` rejects even sizes, non-integers, and `bool` (which Python treats as an
  `int` — `True % 2 == 1` would otherwise sneak through as "size 1").
- A 2-D Gaussian is **separable**: `G(x, y) = g(x) · g(y)`. `np.outer` builds the square
  kernel from one 1-D profile, which is both faster and exactly symmetric by construction
  (`k == k[::-1, ::-1]`, the property the public test checks).
- Normalizing by `k.sum()` after sampling — not by the analytic `2πσ²` — makes the
  *truncated* kernel sum to exactly 1, so a flat region stays flat after filtering.
- Computed in `float64`, cast to `float32` at the end: the sum-to-one is accurate to
  ~1e-7 after the cast, which is what the test's `abs=1e-6` tolerance is for.

---

## `convolve2d_manual`

```python
img = _check_image(image)                # float32, 2-D or (H, W, C)
k = np.asarray(kernel, dtype=np.float32)
kh, kw = k.shape                          # both validated odd
rh, rw = kh // 2, kw // 2
radius = max(rh, rw)
padded = _pad_spatial(img, radius, border_mode)
H, W = img.shape[:2]
flipped = k[::-1, ::-1]                   # <-- true convolution
out = np.zeros(img.shape, dtype=np.float32)
for i in range(kh):
    for j in range(kw):
        w = flipped[i, j]
        if w == 0:
            continue
        y0 = radius - rh + i
        x0 = radius - rw + j
        out += w * padded[y0:y0 + H, x0:x0 + W]
return out.astype(np.float32)
```

- **The loop is over kernel taps, not pixels.** A 7×7 kernel means 49 iterations, each of
  which is one vectorized NumPy multiply-add over a full `H×W` slice. That is how a
  "manual" convolution stays fast (~10 ms for a 25×25 kernel on this image) without
  delegating to `cv2.filter2D` or `scipy.signal.convolve`.
- `_pad_spatial` pads **only the first two axes**; for an `(H, W, C)` image the channel axis
  gets `(0, 0)` padding, so each channel is filtered independently with the same kernel.
  That is the whole multichannel story — no per-channel loop is needed because the slice
  `padded[y0:y0+H, x0:x0+W]` carries the channel axis along for free.
- The slice offset `radius − rh + i` walks the kernel across the padded image. Because the
  kernel was already reversed, tap `(i, j)` of `flipped` multiplies the pixel that sits
  *opposite* to where tap `(i, j)` of `k` would have sat under correlation.
- `if w == 0: continue` is a small optimization that matters for the sparse asymmetric
  test kernel (six of nine taps are zero).
- Everything is `float32` in and out; the caller's array is never written to.

---

## `sharpen_image` and `gaussian_support`

```python
def gaussian_support(sigma):
    return max(3, 2 * int(math.ceil(3.0 * sigma)) + 1)

def sharpen_image(image, amount, sigma):
    if not (amount >= 0): raise ValueError(...)
    if not (sigma > 0):   raise ValueError(...)
    img = _check_image(image)
    blur = convolve2d_manual(img, gaussian_kernel(gaussian_support(sigma), sigma), "reflect")
    return (img + np.float32(amount) * (img - blur)).astype(np.float32)
```

- **Support choice, stated:** `max(3, 2·⌈3σ⌉ + 1)`. ±3σ captures 99.7 % of a Gaussian, so
  the truncation error is negligible; σ = 1.5 → 11×11, σ = 2 → 13×13. Exposing it as a
  named function lets the notebook write the size into the metrics' `condition` strings.
- `not (amount >= 0)` rather than `amount < 0` — the former also rejects `NaN`.
- No clipping. `amount = 0` returns the input bit-for-bit (public test), and the output is
  exactly linear in `amount` — the notebook checks
  `sharpen(2a) − x == 2·(sharpen(a) − x)` to 1.8e-7.
- The blur uses `reflect` unconditionally; that is a design decision, not a parameter, and
  it is the border mode under which edge behavior is least distorted.

---

## The notebook cell logic (`5bdce6fe`)

**The shared metrics helper.** Task 1 is the first experiment, so it defines the function
every later cell uses:

```python
def append_metrics(records, path=METRICS_PATH):
    new = pd.DataFrame(records)
    new = new[np.isfinite(pd.to_numeric(new["value"], errors="coerce"))]
    ...
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    key = ["experiment", "method", "condition", "metric"]
    new = new.drop_duplicates(subset=key, keep="last")
    new.to_csv(path, index=False)
```

- Long-form rows with the six required columns (`experiment, method, condition, metric,
  value, units`) — the schema `validate_submission.py` checks.
- Non-finite values are dropped *before* writing, per the README ("append later
  finite-valued records").
- The drop-duplicates on the four-column key is what makes the CSV **cumulative and
  idempotent**: re-running one cell replaces only its own rows, and a full restart-and-run
  rebuilds the same file. Nothing ever deletes another experiment's rows.

**(a) Proving true convolution.** An asymmetric 3×3 kernel is run through
`convolve2d_manual` and through `cv2.filter2D` twice — once with the kernel flipped (which
turns `filter2D`'s correlation into convolution) and once unflipped:

| reference | max abs error |
|---|---|
| `filter2D(flipped)` | 6e-8 |
| `filter2D(unflipped)` | 0.51 |

Same code, same border mode; the only difference is the flip. That table is the evidence
that the kernel reversal is real and in the right place.

**(b) Gaussian sweeps.** Two sweeps, both at all three border modes:

- σ ∈ {1, 2, 4} with the support tied to σ (7, 13, 25) — what σ does.
- σ = 2 with support ∈ {5, 9, 13, 21} — what truncating the support does at fixed σ.

For each: max/mean error vs `filter2D(flipped)` and vs `cv2.GaussianBlur` (both named in
the printout), `mse_vs_original` (how much the blur changed the image), `edge_energy_ratio`
(mean gradient magnitude relative to the original — an edge-strength proxy), and
`border_band_mae_vs_reflect` (how different the outermost rows are from the `reflect`
result — the border-mode effect isolated to where it lives).

**(c) Multichannel.** One 13×13 σ = 2 pass on `color_scene.png` against `filter2D`:
6.6e-7. Same slice-accumulation path, channel axis carried along.

**(d) Sharpening sweeps.** Amount ∈ {0.5, 1, 2, 4} at σ = 1.5, and σ ∈ {0.75, 3} at
amount 1. Recorded: `max_value`, `min_value`, `overshoot_fraction` (fraction of pixels
outside `[0, 1]` — the halo footprint), `edge_energy_ratio`, and the linearity check.

**The figure.** Two rows of four: the original and the three σ blurs (reflect), then the
original and three sharpening amounts. Sharpened panels are `np.clip`ped **for display
only** — the metrics come from the unclipped arrays.

---

## The measured result

**Reference agreement** — every Gaussian configuration matches `filter2D(flipped)` to
≤ 1.1e-6 and `GaussianBlur` to ≤ 1.2e-6 (float32 accumulation order). Border mode does not
change the error, which confirms `reflect ↔ BORDER_REFLECT_101`, `edge ↔ BORDER_REPLICATE`,
`constant ↔ BORDER_CONSTANT` are the right pairings.

**Smoothing** (reflect):

| σ | support | mse vs original | edge energy ratio |
|---|---|---|---|
| 1 | 7 | 0.0096 | 0.66 |
| 2 | 13 | 0.0255 | 0.27 |
| 4 | 25 | 0.0359 | 0.10 |

Detail loss and edge weakening both scale with σ; at σ = 4 the stripes and checkerboard in
the test pattern are gone (visible in the figure).

**Support at fixed σ = 2** — the 5×5 kernel differs from the full 13×13 by a measurable
amount (it is a Gaussian truncated at ±1σ), while 13 vs 21 is indistinguishable: ±3σ is
enough.

**Border modes** — `constant` raises the border-band MAE to 0.17 (σ = 1) and 0.25 (σ = 4):
zero padding darkens the frame. `edge` differs from `reflect` by ≤ 4e-5 on this image
because the test pattern's borders are already nearly flat.

**Sharpening** (σ = 1.5):

| amount | max value | overshoot fraction | edge energy ratio |
|---|---|---|---|
| 0.5 | 1.07 | 0.056 | 1.40 |
| 1.0 | 1.28 | 0.24 | 1.81 |
| 2.0 | 1.71 | 0.28 | 2.62 |
| 4.0 | 2.57 | 0.30 | 4.23 |

Edge energy grows linearly with `amount` (as the formula says it must); overshoot saturates
at ~30 % of pixels because that is how many pixels sit within the blur support of an edge.
Larger σ at fixed amount (0.75 → 3.0) widens the halo band: overshoot fraction 0.16 → 0.27.

---

## Two concepts examiners love to probe

1. **Why flip the kernel, and how would you prove your code does?** Because convolution is
   defined with the kernel reversed; correlation is not. The proof is an asymmetric kernel:
   your result matches `filter2D` on the *flipped* kernel to float precision and differs
   from the unflipped call by ~0.5. A symmetric kernel would never expose the bug.
2. **What is `reflect` and why does the choice matter?** NumPy `reflect` = OpenCV
   `BORDER_REFLECT_101`: mirror without repeating the edge pixel. It matters twice — once
   for the comparison (mismatched modes look like a convolution bug at the border), and once
   for the result (zero padding leaves a dark frame; the border-band MAE quantifies it).
