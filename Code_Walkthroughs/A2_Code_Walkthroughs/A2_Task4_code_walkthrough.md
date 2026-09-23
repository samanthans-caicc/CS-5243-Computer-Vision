# A2 Task 4 — Code Walkthrough

_Study note (not part of the graded submission). Explains `gaussian_pyramid`,
`laplacian_pyramid`, `reconstruct_laplacian_pyramid`, and the `reduce_image` /
`expand_image` helpers in `A2/src/student_code.py`, plus the Task 4 notebook cell
(`1db8a3c8`)._

---

## Key background: a pyramid is a change of representation, not a compression

A **Gaussian pyramid** is the image blurred and halved, repeatedly: `G0` is the image,
`G1` is `reduce(G0)`, and so on. Each level is a lower-resolution, lower-frequency version
of the one above.

A **Laplacian pyramid** stores the *differences* between consecutive Gaussian levels:
`L_i = G_i − expand(G_{i+1})`. Each `L_i` is a **band-pass** image — the detail that
exists at level `i`'s resolution but not at level `i+1`'s. The coarsest level is kept as
the Gaussian `G_{n−1}` itself, because there is nothing below it to subtract.

The whole point is that the Laplacian pyramid is **invertible**:

```
G_{n-1}                      (start with the coarsest)
G_{i} = expand(G_{i+1}) + L_i     (add back one band at a time)
```

Nothing is lost — the reconstruction should match the original to floating-point
precision. That is the *mathematical correctness* check. Separately, a pyramid with one
band zeroed out still reconstructs into a picture that *looks* right — that is *visual
plausibility*, and the task asks you to tell the two apart with numbers.

Two practical facts drive the code:

- **Odd sizes.** `257 → 129 → 65 → 33` under ceiling-halving `(n + 1) // 2`. Going back up
  is *not* `× 2` (`33 · 2 = 66 ≠ 65`), so `expand` must be told the exact target shape it
  is reconstructing. `cv2.pyrUp(..., dstsize=)` accepts any size within ±2 of double, which
  is exactly what ceiling-halving needs.
- **`cv2.pyrDown` / `cv2.pyrUp` are approved primitives** — they do one Gaussian-prefiltered
  halving / one Gaussian-interpolated doubling. The *pyramid logic* — looping, subtracting,
  ordering, reconstructing — is the student's.

---

## `reduce_image` / `expand_image`

```python
def reduce_image(image):
    img = _check_image(image)
    H, W = img.shape[:2]
    out = cv2.pyrDown(img, dstsize=((W + 1) // 2, (H + 1) // 2))
    return _restore_channels(out, img).astype(np.float32)

def expand_image(image, target_shape):
    img = _check_image(image)
    H, W = int(target_shape[0]), int(target_shape[1])
    out = cv2.pyrUp(img, dstsize=(W, H))
    return _restore_channels(out, img).astype(np.float32)
```

- `dstsize` is passed **explicitly** in both directions. `pyrDown`'s default is already
  ceiling-halving, but stating it makes the contract visible; `pyrUp` has no way to guess
  65 from 33, so the target shape is mandatory.
- OpenCV's `Size` is `(width, height)` — the reverse of NumPy's `shape`. Both helpers do
  the swap in one place so no caller has to remember it.
- `_restore_channels` puts back a trailing singleton channel axis if the input had one
  (`(H, W, 1)` → OpenCV returns `(H, W)`). Ordinary RGB and grayscale never hit it.

## `gaussian_pyramid`

```python
levels = _check_levels(levels)
current = _check_image(image)
pyramid = [current]
for _ in range(levels - 1):
    current = reduce_image(current)
    pyramid.append(current)
return pyramid
```

Exactly `levels` arrays, finest first, each one `reduce` of the previous. `levels = 1`
returns just the input.

## `laplacian_pyramid`

```python
gauss = gaussian_pyramid(image, levels)
pyramid = []
for finer, coarser in zip(gauss[:-1], gauss[1:]):
    pyramid.append((finer - expand_image(coarser, finer.shape)).astype(np.float32))
pyramid.append(gauss[-1])
return pyramid
```

- `zip(gauss[:-1], gauss[1:])` pairs each level with the next-coarser one — the classic
  idiom for "consecutive pairs".
- `expand_image(coarser, finer.shape)` is where odd sizes are handled: the expansion is
  told to produce exactly `finer`'s shape, so the subtraction always lines up.
- The last element is the coarsest **Gaussian**, appended unmodified. Mixing band-pass
  levels with one low-pass level in the same list is the standard Burt–Adelson layout.

## `reconstruct_laplacian_pyramid`

```python
current = np.asarray(pyramid[-1], dtype=np.float32)
for level in reversed(pyramid[:-1]):
    current = (expand_image(current, level.shape) + level).astype(np.float32)
return current
```

The decomposition loop run backwards: start at the coarsest, expand to the *next level's
recorded shape*, add that band. Because the shapes come from the pyramid itself, a pyramid
of any odd/even mix reconstructs to the original size without any arithmetic.

---

## The notebook cell logic (`1db8a3c8`)

**Build and check.** `color_scene.png` (257×385×3), `levels = 4` — the values the task
prescribes for the saved artifacts:

```python
gauss = sc.gaussian_pyramid(scene, 4)
lap   = sc.laplacian_pyramid(scene, 4)
recon = sc.reconstruct_laplacian_pyramid(lap)
err   = np.abs(recon - scene)
```

**`pyramid_metrics.json`** carries every key the README requires — `levels`,
`level_shapes` (finest first, `[H, W, C]`), `reconstruction.max_abs_error`,
`reconstruction.mean_abs_error`, `reconstruction.dtype` — plus `source_image`
(`"color_scene.png"`, required by the updated README) and a few diagnostics (Laplacian
shapes, PSNR, which primitives were used). `a2_tools.save_json` writes it.

**Sweeps into the CSV:**

- Reconstruction error for `levels ∈ {2..6}` on both `color_scene.png` and the odd-sized
  grayscale `test_pattern_gray.png` (255×383 → 128 → 64 → 32 → 16 → 8).
- **`reconstruction_without_L0`** — the finest band replaced by zeros, then reconstructed.
  Recorded with max error, PSNR, and SSIM. This is the "visually plausible but wrong"
  control.
- RMS energy of each Laplacian band — what each level actually carries.

**Two figures.** `pyramid_visualization.png`: Gaussian levels on top, Laplacian levels
below. Band-pass levels are rescaled for display by `show_laplacian()` — `0.5 + L / (2·max|L|)`
— so zero maps to mid-grey and negative detail shows as darker than grey; without this
they would render as near-black. `pyramid_reconstruction.png`: original, reconstruction,
the per-pixel error map (`magma`, with a colour bar), and the no-L0 reconstruction beside
it.

---

## The measured result

| check | max abs error | mean abs error |
|---|---|---|
| `color_scene.png`, levels 2–6 | 1.5e-8 | 1.9e-11 |
| `test_pattern_gray.png`, levels 2–6 | 6.0e-8 | — |
| reconstruction **without L0** | **0.463** | (PSNR 28.8 dB, SSIM 0.94) |

- 1.5e-8 is one float32 ulp at 1.0. The reconstruction is exact up to rounding; the
  number does not grow with `levels`, which shows the expand-to-recorded-shape logic is
  right at every odd/even transition (257→129→65→33→17→9 all round-trip).
- Dropping `L0` leaves a reconstruction with PSNR 28.8 dB and SSIM 0.94 — a picture most
  people would accept as the original — while the max error is **0.46**, almost half the
  dynamic range, on the fine edges that `L0` carried. That pair of numbers is the
  distinction the analysis prompt asks for: a high PSNR is *plausibility*; a max error at
  the precision floor is *correctness*.
- Band energies: `L0`–`L2` are all ~0.035–0.039 RMS (detail spread across scales), `L3`
  (the Gaussian residual) is 0.43 — it holds the DC/low-frequency content, i.e. most of
  the signal energy. Band-pass levels are small and signed; that is why they need rescaling
  to be seen.

Shapes, for reference: `[257,385,3] → [129,193,3] → [65,97,3] → [33,49,3]`.

---

## Two concepts examiners love to probe

1. **Why must `expand` be told the target shape?** Because ceiling-halving is not
   invertible by doubling: 65 halves to 33, and 33 doubles to 66. The Laplacian
   subtraction `G_i − expand(G_{i+1})` only works if `expand` produces `G_i`'s exact shape,
   so every level records it and `pyrUp(dstsize=)` honours it.
2. **What does a small reconstruction error prove, and what does it not?** It proves the
   decomposition and reconstruction are mathematically inverse — no band was dropped or
   misaligned. It says nothing about whether the *pyramid itself* is a good
   representation; for that you look at the bands (do they separate scales?) and at what
   happens when you tamper with one (the no-L0 control).
