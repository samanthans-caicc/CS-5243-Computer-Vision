# A2 Task 6 — Code Walkthrough

_Study note (not part of the graded submission). Explains the Task 6 notebook cell
(`144899a7`). No new graded function — this cell re-uses `gaussian_pyramid`,
`laplacian_pyramid`, `expand_image`, and `laplacian_pyramid_blend` from Tasks 4 and 5 to
make the blending *process* visible._

---

## Key background: what you are supposed to be able to see

`laplacian_pyramid_blend` returns one image. Task 6 opens it up and lays every intermediate
on the page so that seven claims become inspectable:

1. **Gaussian pyramid of each source** — the same image at four resolutions; fine texture
   disappears level by level.
2. **Laplacian pyramid of each source** — the band-pass residuals. Near zero (mid-grey after
   rescaling) in flat regions; signed edge detail at each scale; the coarsest level is the
   Gaussian residual and looks like a small blurry image.
3. **Gaussian pyramid of the mask** — the weight at each level. The hard `blend3` mask stays
   hard-edged at level 0 and grows a soft ramp at each halving; the soft `blend` mask starts
   soft and gets softer.
4. **Blended Laplacian pyramid** — per level, `(1 − Gm)·La + Gm·Lb`. Left of the seam looks
   like `La`, right of it like `Lb`, and the handover is wide at coarse levels and narrow at
   fine ones.
5. **Progressive reconstruction** — start from the coarsest blended level and add one band
   at a time. The first stage is a tiny blurry blend with no seam; each stage adds sharper
   detail and the seam never appears.

The last row is the demonstration of *why* the method works: a seam is a high-frequency
event, and at no stage is a high-frequency band ever blended with a hard weight.

---

## The notebook cell logic (`144899a7`)

**Two pairs, as the task requires — one synthetic, one real:**

```python
decomp_pairs = {
    "synthetic: blend_left/right + blend_mask (soft vertical)": (...),
    "real photo: blend3_left/right + blend3_mask (hard vertical)": (...),
}
DECOMP_LEVELS = 4
```

**`decompose_blend()`** — the blend function's body, unrolled so every intermediate is kept:

```python
ga, gb = sc.gaussian_pyramid(a, levels), sc.gaussian_pyramid(b, levels)
la, lb = sc.laplacian_pyramid(a, levels), sc.laplacian_pyramid(b, levels)
gm     = sc.gaussian_pyramid(m, levels)
blended = [((1 - w[..., None]) * x + w[..., None] * y) for x, y, w in zip(la, lb, gm)]
progressive = [blended[-1]]
for level in reversed(blended[:-1]):
    progressive.append(sc.expand_image(progressive[-1], level.shape) + level)
final = sc.laplacian_pyramid_blend(a, b, m, levels)
assert np.max(np.abs(progressive[-1] - final)) < 1e-5
```

- Every array comes from the Task 4/5 functions; nothing is re-implemented. `expand_image`
  is the same helper `reconstruct_laplacian_pyramid` uses internally, called here one step
  at a time so each stage can be captured.
- `progressive` is built coarse → fine: `[blended[-1]]` is stage 0 (the 49×33 base), and
  each iteration expands the running result to the next level's *recorded shape* and adds
  that band — the reconstruction loop, made visible.
- **The `assert`** is the cell's own correctness check: the staged path must reproduce the
  graded function to 1e-5. If someone later edits `laplacian_pyramid_blend` so that it no
  longer matches this decomposition, the notebook stops here rather than silently drawing a
  picture of a different algorithm.

**Display rescaling.** Band-pass levels (Laplacian and blended rows, all but the coarsest)
go through `show_laplacian()` from Task 4 — `0.5 + L / (2·max|L|)` — so zero is mid-grey
and the sign of the detail is visible. Gaussian levels, mask levels, the coarsest
Laplacian level, and the progressive stages are ordinary `[0, 1]` images and are just
clipped.

**Layout.** One `7 × 8` grid: seven rows (the five categories above, with A and B each
getting their own Gaussian and Laplacian rows), four columns per pair, the two pairs side
by side. Each row's first panel carries the row title; the top-left panel of each block
carries the pair label in brackets. Every panel is titled with its level (or stage) and its
`width×height`, so the halving is legible without reading the code. `imshow` draws every
level in the same-size axes, so a 49×33 level is upscaled for display — the title is what
tells you its true size.

Saved at `dpi=130` to `outputs/figures/pyramid_decomposition.png` (2067×1741, above the
600×600 minimum).

---

## What the figure shows

**Synthetic pair (left block).** The Laplacian levels of the flat-coloured shapes are
almost entirely mid-grey with thin outlines — synthetic images have detail *only* at edges.
The soft mask's Gaussian pyramid barely changes across levels because it was already
smooth. The blended Laplacian levels show the left half's outlines and the right half's
stripes with a gentle handover; the progressive reconstruction is seamless from stage 0.

**Real-photo pair (right block).** The Laplacian levels are full of fine texture (orange
peel at `L0`, dimples at `L1`, the highlight blob at `L2`). The hard mask is a step at
level 0, a 2-px ramp at level 1, and a visibly soft ramp by level 3 — that widening is the
mechanism. In the blended Laplacian row, apple texture on the left meets orange texture on
the right at every level, but at `L3` (the low-pass residual) the apple's red and the
orange's orange cross-fade over ~10 px of the 49-px-wide level, i.e. ~80 px at full
resolution. The progressive reconstruction shows a blurry two-tone fruit at stage 0 that
acquires texture stage by stage, and the seam that direct blending would show is never
introduced.

---

## Two concepts examiners love to probe

1. **What does one Laplacian level represent, and why is it near-black before rescaling?**
   The detail present at that resolution and absent at the next coarser one — a band of
   spatial frequencies. It is signed and small (RMS ≈ 0.035 on `color_scene.png`) because
   most pixels sit in regions that the next level already reproduces; only edges leave a
   residual. Mapping zero to mid-grey is what makes the band visible.
2. **How does this figure connect to the Task 4 reconstruction check?** The last row *is*
   `reconstruct_laplacian_pyramid` executed one step at a time on the blended pyramid; the
   `assert` ties its final stage to `laplacian_pyramid_blend` at 1e-5, and Task 4 tied the
   same expand-and-add loop to the original image at 1.5e-8. The blend is exact
   reconstruction of a pyramid whose bands were mixed — nothing about the reconstruction
   changed, only the bands did.
