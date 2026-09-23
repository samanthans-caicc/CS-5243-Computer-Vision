# A2 Task 5 — Code Walkthrough

_Study note (not part of the graded submission). Explains `laplacian_pyramid_blend` in
`A2/src/student_code.py` and the Task 5 notebook cell (`728dd7f7`)._

---

## Key background: blend each frequency band over a distance matched to its scale

**Direct blending** (splicing) is `out = (1 − m)·A + m·B` with the mask `m` as given. With
a hard mask the seam is a one-pixel step between two images that differ in colour and
texture; with a soft mask the transition is a fixed width for every kind of content — too
wide for fine texture (it ghosts), too narrow for broad colour (it shows).

**Laplacian pyramid blending** (Burt & Adelson, 1983) does the same weighted sum, but
**per band of the Laplacian pyramid**, and with the mask's **Gaussian pyramid** as the
weight at each level:

```
blended_i = (1 − Gm_i) · La_i + Gm_i · Lb_i
```

Because the mask is blurred and halved along with the images, a coarse band (broad colour)
is blended across a wide, smooth transition, while a fine band (texture) is blended across a
narrow one. Reconstructing the blended pyramid gives one image with no visible seam at any
scale. That is why mask 0 → `A`, mask 1 → `B`, and the transition width scales
*automatically* with content — no hand-tuned feathering.

The evidence question: *how do you measure a seam?* The notebook uses gradient energy in a
band around the mask transition, compared to what the sources themselves have there. A
seam adds gradient that neither source has; a ratio near 1.0 means "no more edges than the
photos already contained."

---

## `laplacian_pyramid_blend`

```python
a = _check_image(image_a)
b = _check_image(image_b)
m = np.asarray(mask, dtype=np.float32)
if a.shape != b.shape:            raise ValueError(...)
if m.shape[:2] != a.shape[:2]:    raise ValueError(...)
if m.ndim == 3 and m.shape[2] == 1: m = m[..., 0]
if m.ndim != 2:                   raise ValueError(...)

lap_a   = laplacian_pyramid(a, levels)
lap_b   = laplacian_pyramid(b, levels)
gauss_m = gaussian_pyramid(m, levels)
blended = []
for la, lb, gm in zip(lap_a, lap_b, gauss_m):
    weight = gm[..., None] if la.ndim == 3 else gm
    blended.append(((1.0 - weight) * la + weight * lb).astype(np.float32))
return reconstruct_laplacian_pyramid(blended)
```

- **Three pyramids, one loop.** Two Laplacian pyramids for the images and one *Gaussian*
  pyramid for the mask — the mask must be low-passed, not band-passed, because it is a
  weight, not a signal. Getting this wrong (a Laplacian mask pyramid) produces garbage.
- `gm[..., None]` adds a channel axis so a `(h, w)` mask level broadcasts against an
  `(h, w, 3)` image level. The `if` keeps grayscale blends working too.
- The blend is the same `(1 − w)·a + w·b` formula at every level; only `w` changes. The
  contract "mask 0 selects `image_a`" falls out algebraically: a zero mask gives a zero
  Gaussian pyramid, so `blended == lap_a`, and reconstruction returns `a` exactly (the
  public test).
- The mask's spatial shape must match the images; the function tolerates a stray `(H, W, 1)`
  mask by squeezing it, and rejects anything else.
- No clipping on output. A blend of two `[0, 1]` images stays in `[0, 1]` up to rounding;
  the notebook clips for display only.

---

## The notebook cell logic (`728dd7f7`)

**Pairs.** Three of the dataset's four aligned pairs, chosen to span mask geometries:

| key | files | mask |
|---|---|---|
| `blend3_hard_vertical` | apple / orange photos | hard vertical seam |
| `blend4_hard_circular` | apple / orange close-ups | hard circular seam |
| `blend2_diagonal` | synthetic scenes | soft diagonal seam |

Two *different image pairs* are required; three are used, and the two real-photo pairs
carry the hard masks that make the method difference obvious.

**Seam metric.**

```python
def seam_band(mask, width=7):
    gy, gx = np.gradient(mask)
    edge = (np.sqrt(gx**2 + gy**2) > 0.01).astype(np.uint8)
    return cv2.dilate(edge, np.ones((width, width))).astype(bool)
```

Pixels where the mask *changes*, dilated by 7 px — the band where a seam would live. For a
hard mask that is a 7-px strip along the boundary; for the soft diagonal mask it is the
whole feathered ramp. `seam_metrics()` then records:

- `seam_band_gradient` — mean luma gradient magnitude inside the band.
- `seam_band_gradient_ratio_vs_sources` — the same, divided by the average of the two
  *source* images' gradient energy in that band. **This is the headline number**: 1.0 means
  the blend has no more edge energy at the seam than the photographs already had.
- `seam_band_fraction` — what fraction of the image the band covers (context for the above).

**Per pair:** direct blend, Laplacian blend at `levels = 5`, and a level sweep
`{1, 2, 3, 7}`. `levels = 1` is a Laplacian pyramid with only the base level — it is
*exactly* direct blending, and the metrics confirm it.

**Deliberately poor configurations** on `blend3`:

- `np.roll(mask, 40, axis=1)` with the wrapped columns patched — the mask boundary is 40 px
  right of where the apple meets the orange.
- `levels = 2` with the correct mask — too few levels for the transition to widen.

**Figure.** One row per pair: `image_a`, `image_b`, mask, direct, Laplacian (L = 5). A
fourth row of the poor configurations: shifted mask, direct with it, Laplacian with it,
Laplacian L = 2, and Laplacian L = 1 (labelled `== direct`).

---

## The measured result

Seam-band gradient ratio vs sources (1.0 = no added edge energy):

| pair | direct | L = 2 | L = 3 | L = 5 | L = 7 |
|---|---|---|---|---|---|
| blend3 hard vertical | 2.76 | 2.39 | 1.91 | **1.15** | 1.03 |
| blend4 hard circular | 2.65 | 2.12 | 1.72 | **1.07** | 1.00 |
| blend2 diagonal (soft) | 1.08 | 1.08 | 1.08 | **1.06** | 1.05 |
| blend3, mask shifted 40 px | 3.12 | — | — | 1.16 | — |

- **Hard masks: direct splicing nearly triples the edge energy at the seam** (2.76×,
  2.65×); pyramid blending at 5 levels brings it back to within 7–15 % of the sources'
  own texture, and at 7 levels to parity. The seam is gone in the numbers, not just in the
  picture.
- **The level sweep is monotone**, and `L = 1` reproduces the direct-blend value exactly —
  a built-in sanity check that the two methods differ only by the pyramid.
- **The soft diagonal mask barely benefits** (1.08 → 1.06): its feathered ramp already
  spreads the transition, so direct blending is nearly seamless there. Pyramid blending
  matters most exactly where the mask is hardest.
- **The shifted mask is the worst case for direct blending** (3.12×) because the boundary
  now cuts through the middle of the orange's texture instead of the fruit edge. Pyramid
  blending still smooths the *seam* (1.16×) — but the figure shows what the metric cannot:
  a smooth transition in the wrong place is still wrong. The mask's *placement* is a design
  decision the algorithm cannot fix.
- **Too few levels** (L = 2, 2.39×) leaves a visible seam because the coarsest band —
  where the colour difference between apple and orange lives — is still blended over a
  transition only ~2 px wide.

---

## Two concepts examiners love to probe

1. **Why does per-level blending remove a seam that pixel blending cannot?** Low
   frequencies (colour, shading) are blended across a wide transition because the mask has
   been blurred many times by that level; high frequencies (texture) across a narrow one.
   Each band gets the transition width appropriate to its wavelength. A single pixel-level
   mask has to pick one width for everything.
2. **Which pyramid does the mask get, and why not the same one as the images?** The
   *Gaussian* pyramid — the mask is a weight, and each level needs the full low-passed
   weight, not the band-pass difference between levels. A Laplacian mask level is near zero
   almost everywhere and would select neither image.
