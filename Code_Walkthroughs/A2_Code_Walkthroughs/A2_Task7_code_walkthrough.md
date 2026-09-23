# A2 Task 7 — Code Walkthrough

_Study note (not part of the graded submission). Explains the Task 7 failure-analysis
notebook cell (`07b38336`). It re-uses `sharpen_image`, `median_filter_manual`, and the
Task 1/2 helpers; no new graded function._

---

## Key background: a controlled failure has four parts

The rubric wants, for each failure: **identify** it, **show** evidence, **explain** the
mechanism, **mitigate** it, and **verify** the mitigation with the same measurement. The
cell is organized so every one of those steps is a line of code with a number attached.

The two failures are from different topics, as required:

**Failure 1 — sharpening halos.** Unsharp masking adds `amount · (img − blur)`. At a step
edge, `img − blur` is a positive lobe on the bright side and a negative lobe on the dark
side, each about as wide as the blur support. Multiply by a large `amount` and the bright
side overshoots past 1 and the dark side undershoots below 0 — a bright/dark fringe, the
halo. The mechanism is linear, so the halo amplitude scales with `amount` and its width
with σ.

**Failure 2 — a median window wider than the structure.** A median replaces each pixel by
the middle value of its window. That is only "robust" while the pixel's own structure fills
more than half the window. The test pattern's stripes are ~4 px wide; a 7×7 window
straddling a stripe contains more background than stripe, so the stripe *is* the outlier
and gets removed. The filter then scores *worse* than the noisy input it was meant to fix.

---

## The notebook cell logic (`07b38336`)

### Failure 1

```python
FAIL_AMOUNT, FAIL_SIGMA = 6.0, 2.0
FIX_AMOUNT,  FIX_SIGMA  = 1.0, 1.0
sharp_fail = sc.sharpen_image(gray, FAIL_AMOUNT, FAIL_SIGMA)
sharp_fix  = sc.sharpen_image(gray, FIX_AMOUNT,  FIX_SIGMA)
```

The approved fallback: an excessive amount (6, three times the largest value in the Task 1
sweep) at a wide σ, then a reduced configuration.

A second, *reformulated* mitigation keeps the aggressive amount but clamps the result to
the local range of the original:

```python
support = sc.gaussian_support(FAIL_SIGMA)                    # 13
kernel_box = np.ones((support, support), np.uint8)
local_min, local_max = cv2.erode(gray, kernel_box), cv2.dilate(gray, kernel_box)
sharp_clamped = np.clip(sc.sharpen_image(gray, 6.0, 2.0), local_min, local_max)
```

`erode`/`dilate` with a box the size of the blur support give, per pixel, the minimum and
maximum of the original within that window. Clipping the sharpened value into `[min, max]`
means it can never exceed any value that already existed nearby — overshoot is impossible
by construction. (OpenCV morphology is a library call, but it is a *mitigation*, not a
required implementation.)

`halo_stats()` records for each of the three: `overshoot_fraction` (pixels outside
`[0, 1]`), `max_value`, `min_value`, `halo_amplitude` (how far past the range the worst
pixel goes), `edge_energy_ratio`, and `ssim_vs_original`.

### Failure 2

```python
imp = noisy["impulse"]                          # Task 2's own p = 0.07 observation
med_fail = sc.median_filter_manual(imp, 7)
med_fix  = sc.median_filter_manual(imp, 3)
impulse_mask = (imp <= 0.0) | (imp >= 1.0)
med_switch = np.where(impulse_mask, med_fix, imp)
```

- `med_fail` is the Task 2 sweep's worst setting, reproduced.
- `med_fix` is the plain mitigation — a window narrower than the structure.
- `med_switch` is the reformulated one — a **switching median**: only pixels that are
  exactly 0 or 1 (the only values an impulse can take) are replaced by the 3×3 median;
  every other pixel is left untouched. With `p = 0.07`, 7.1 % of pixels are replaced
  (`replaced_fraction`), and the remaining 93 % keep their exact clean value.

`denoise_stats()` records PSNR, SSIM, RMSE, and `edge_energy_ratio_vs_clean` against the
clean reference, using the same `quality()` helper as Task 2.

### Figure and CSV

`failure_analysis.png` (2 × 4): row 1 — a crop of the original, the halo failure, the
reduced-amount mitigation, and an **intensity profile along row 40** (which crosses the
dark block's step edge and the checkerboard) with the original, failure, mitigation, and
clamped curves overlaid on `y = 0` / `y = 1` guide lines. Row 2 — noisy input, median
k = 7, switching median, clean reference.

All records go in with `experiment = "failure"` and `method` naming the failure
(`sharpen_halo`, `median_detail_loss`); `condition` starts with `failure_` or
`mitigated_` so the CSV reads as identify → mitigate → verify.

---

## The measured result

**Failure 1 — halos:**

| configuration | overshoot fraction | max | min | halo amplitude | edge energy | SSIM |
|---|---|---|---|---|---|---|
| amount 6, σ 2 (**failure**) | 0.323 | 3.74 | −3.14 | 3.14 | 6.64× | 0.857 |
| amount 1, σ 1 (mitigated) | 0.165 | 1.20 | −0.29 | 0.29 | 1.56× | 0.967 |
| amount 6, σ 2, clamped | **0.000** | 0.90 | 0.10 | **0.00** | 1.04× | 0.966 |

- At amount 6 the worst pixel is at 3.7 — nearly four times white — and a third of the
  image is outside the displayable range. The profile plot shows the checkerboard swinging
  between +3 and −2.
- Reducing the amount cuts the halo amplitude by 11× and lifts SSIM to 0.97, but a 0.29
  overshoot is still there (unsharp masking *always* overshoots at a step; the question is
  how much).
- Clamping removes overshoot entirely. But note `edge_energy 1.04×` — on this synthetic
  image the steps are already perfectly sharp, so once overshoot is forbidden there is
  almost nothing left for sharpening to do. The clamp is honest about that: it is a
  halo-free sharpener, not a free lunch.

**Failure 2 — median window:**

| configuration | PSNR (dB) | SSIM | edge energy vs clean |
|---|---|---|---|
| noisy input, p = 0.07 | 16.1 | 0.378 | 1.83× |
| median k = 7 (**failure**) | **11.3** | 0.665 | 0.90× |
| median k = 3 (mitigated) | 25.2 | 0.980 | 0.99× |
| switching median k = 3 | **30.9** | **0.995** | 1.00× |

- The failure is *worse than doing nothing* by 4.8 dB — the stripes and checkerboard are
  gone, replaced by grey (visible in the figure), and the edge energy drops below the
  clean image's.
- k = 3 fixes it (+14 dB over the failure) because a 3×3 window straddling a 4-px stripe
  is still majority-stripe.
- The switching median adds another 5.7 dB and reaches SSIM 0.995: the 93 % of pixels that
  were never corrupted are returned *exactly*, and the median only has to be right on the
  7 % that were. Its edge energy ratio of 1.00 says it neither blurred nor sharpened
  anything.

---

## Two concepts examiners love to probe

1. **Why does unsharp masking overshoot, and why does clipping the *output* not fix it?**
   Because `img − blur` has lobes of both signs at every edge, scaled by `amount`. Clipping
   to `[0, 1]` after the fact hides the fringe on a white/black edge but does nothing for a
   grey/grey edge, where the halo lands inside the range. The local-min/max clamp fixes the
   general case because it bounds each pixel by its *own* neighbourhood, not by the global
   range.
2. **When does a median filter stop being robust?** When the structure you want to keep
   occupies fewer than half the window's pixels — then the structure is the minority and
   the median votes it out. Window size has to be chosen relative to the smallest feature
   you need to preserve, not just to the noise density.
