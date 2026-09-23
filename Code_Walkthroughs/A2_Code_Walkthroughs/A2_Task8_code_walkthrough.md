# A2 Task 8 (Extension) — Code Walkthrough

_Study note (not part of the graded submission). Explains the Task 8 extension notebook cell
(`779ac7f6`): frequency-domain filtering compared against the Task 1 spatial-convolution
baseline. Principal artifact: `outputs/extension/frequency_domain_filtering.png`._

---

## Key background: the convolution theorem, and what it does not promise

**Convolution in space is multiplication in frequency:**

```
F{ img ∗ k } = F{img} · F{k}
```

So a Gaussian blur can be done as `ifft2( fft2(img) · H )` where `H` is the kernel's
frequency response. For a Gaussian with spatial std σ, `H` is itself a Gaussian with
frequency std `1 / (2πσ)` cycles per pixel — smooth, positive, monotone.

Three things the FFT route makes cheap that the spatial route does not:

- **Cost is independent of kernel size.** Spatial convolution is `O(H·W·k²)`; FFT is
  `O(H·W·log(H·W))` regardless of σ.
- **Filters that have no compact spatial kernel.** An *ideal* low-pass (`H = 1` inside a
  radius, `0` outside) is one line in frequency; in space it is a `sinc` that never decays.
- **Filter design by shape.** Butterworth `H = 1 / (1 + (f/f_c)^(2n))` is a knob between
  the Gaussian's gentle roll-off and the ideal filter's brick wall.

Two things it does *not* give for free:

- **Boundaries are circular.** `fft2` treats the image as periodic; the top row's
  neighbour is the bottom row. To match a spatial `reflect` result you must pad by
  reflection *before* the FFT and crop after.
- **Sharp cut-offs ring.** Multiplying by a brick wall in frequency is convolving by a
  `sinc` in space, and a `sinc` has negative lobes: the filtered image overshoots near
  edges (Gibbs phenomenon). The Gaussian's `H` has no sharp corner, so no ringing.

The extension's question: *does frequency-domain Gaussian filtering reproduce the Task 1
convolution, and what changes when the filter shape is not Gaussian?*

---

## The notebook cell logic (`779ac7f6`)

**Source and baseline.** `difficult_high_frequency.png` (193×319, the dataset's
high-frequency stress case), σ = 3, support 19 (`gaussian_support`). The baseline is
`sc.convolve2d_manual(hf, kernel, "reflect")` — the student's own Task 1 convolution,
timed.

**The FFT path with matched borders:**

```python
def fft_filter(img, transfer, pad):
    padded = np.pad(img, pad, mode="reflect")
    out = np.real(np.fft.ifft2(np.fft.fft2(padded) * transfer))
    return out[pad:pad + H, pad:pad + W]
```

- `pad = 19` (one kernel width) of `reflect` padding on all sides, so the periodic
  wrap-around happens 19 px outside the image and never reaches it. Crop back afterwards.
- `np.real` discards the ~1e-16 imaginary residue of a real signal round-tripped through a
  complex FFT.

**Four transfer functions**, all on the padded shape:

```python
transfers = {
    "fft_gaussian_kernel":   kernel_transfer(ext_kernel, padded_shape),     # FFT of the actual 19×19 kernel
    "fft_gaussian_analytic": exp(-(f**2) / (2 * cutoff**2)),                # closed-form, cutoff = 1/(2πσ)
    "fft_ideal_lowpass":     (f <= cutoff),                                 # brick wall
    "fft_butterworth_n2":    1 / (1 + (f / cutoff)**4),                     # order-2 Butterworth
}
```

`kernel_transfer()` zero-pads the spatial kernel to the padded image size and `np.roll`s it
so its centre sits at index `(0, 0)` — that is what makes the FFT product equal *circular
convolution with the kernel centred*, not shifted by half a kernel. `radial_frequency()`
builds `√(fx² + fy²)` from `np.fft.fftfreq` (cycles/pixel, unshifted layout to match
`fft2`'s output).

**Metrics per method** (`experiment = "extension"`): `max_abs_error_vs_spatial_baseline`,
`mean_abs_error_vs_spatial_baseline`, `overshoot_fraction` and `ringing_max_abs_overshoot`
(pixels / distance outside the *input's* own `[min, max]` — a low-pass filter can never
legitimately produce a value the input did not have, so any excess is ringing), `runtime`,
and `edge_energy_ratio`.

**Figure.** Top row: the input and the five filtered results, each titled with its max
difference from the spatial baseline. Bottom row: the input's log-magnitude spectrum, the
spatial kernel, and `|H(u, v)|` (fft-shifted so DC is in the centre) for each frequency
filter. Saved to `outputs/extension/` — the one declared extension artifact.

---

## The measured result

| method | max |Δ| vs spatial | overshoot fraction | ringing amplitude | runtime (ms) |
|---|---|---|---|---|---|
| spatial Gaussian (baseline, Task 1) | 0 | 0 | 0 | 11.0 |
| FFT of the same 19×19 kernel | **7.2e-7** | 0 | 0 | 3.9 |
| FFT analytic Gaussian | 8.3e-4 | 0 | 0 | 3.3 |
| FFT ideal low-pass | 0.146 | **0.0037** | **0.048** | 3.2 |
| FFT Butterworth n = 2 | 0.063 | 0 | 0 | 3.3 |

- **The convolution theorem holds to float precision.** FFT-filtering with the *same
  truncated kernel* matches the spatial convolution to 7e-7 — the reflect-pad-and-crop
  trick made the boundaries agree too, or the error would be ~0.1 at the frame.
- **Analytic vs sampled Gaussian differ by 8e-4.** The analytic `H` is the response of an
  *infinite* Gaussian; the 19×19 kernel is truncated at ±3σ and renormalized. Small, but
  real — and a good example of why "Gaussian filter" needs a support convention (Task 1).
- **The ideal low-pass rings.** 0.37 % of pixels land outside the input's own range, by up
  to 0.048 — the `sinc` sidelobes. Its max difference from the Gaussian result (0.146) is
  mostly this ringing plus the sharper cut-off keeping more mid-frequency detail (edge
  energy 0.145 vs 0.124).
- **Butterworth sits between them:** no overshoot (its `H` is smooth), 0.063 from the
  Gaussian because its roll-off is steeper.
- **Speed:** ~3× faster than the manual spatial convolution at σ = 3, and the gap grows
  with σ (spatial cost ∝ k², FFT cost constant). At a 7×7 kernel the ordering would
  reverse.

---

## Two concepts examiners love to probe

1. **Why pad by reflection before the FFT?** Because the DFT is circular: without padding,
   the blur at the top edge mixes in the bottom edge. Padding by one kernel width with the
   same border rule the spatial filter uses (`reflect`) makes the two results agree at the
   frame — that is the difference between a 7e-7 match and a visible dark band.
2. **Why does an ideal low-pass filter ring and a Gaussian does not?** A discontinuity in
   frequency is a `sinc` in space, and the `sinc`'s negative sidelobes produce over- and
   undershoot at every edge (Gibbs). The Gaussian's transform is another Gaussian —
   smooth, non-negative everywhere — so its spatial kernel has no negative lobes and the
   output stays inside the input's range. `ringing_max_abs_overshoot` (0.048 vs 0.000) is
   the measurement.
