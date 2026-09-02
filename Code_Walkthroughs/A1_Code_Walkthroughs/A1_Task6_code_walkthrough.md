# A1 Task 6 — Code Walkthrough

_Study note (not part of the graded submission). Explains the extension notebook cell
(`f390eecc`): does JPEG compression break the Experiment 2 color rule?_

---

## Key background: what "one controlled comparison" has to look like

The extension is only 4 points and the rubric is explicit about scope: **one question, one
controlled comparison, one principal saved output, one conclusion.** The temptation is to
build something impressive; the actual grading criterion is whether the comparison isolates
a single variable and whether the conclusion follows from a measurement.

So the design is deliberately narrow:

- **Question.** Experiment 2 showed the HSV rule is sensitive to the capture condition.
  Storage is also a capture condition. How much JPEG compression can the image survive
  before the *same* rule stops selecting the *same* target?
- **The one variable.** JPEG quality factor. Everything else is pinned: the same source
  image, the same rule constants (`RULE_LOWER`/`RULE_UPPER` from Experiment 2), the same
  target ROI, the same metrics, the same decode path.
- **The control condition.** The lossless PNG. Without it, "coverage 0.95 at q10" is a
  number with nothing to be compared against.
- **The reference for scale.** Experiment 2's dim and warm results, drawn on the chart. A
  result of "compression barely matters" is only interesting next to something that
  mattered a lot.

Reusing the rule and the ROI is not laziness — it is what makes this an extension of the
experiment rather than a second, unrelated one.

---

## The notebook cell logic (`f390eecc`)

**The shared scorer:**

```python
def rule_scores(rgb_image):
    """Selected fraction, target coverage, and precision for the Experiment 2 rule."""
    hsv = cv2.cvtColor(sc.bgr_to_rgb(rgb_image), cv2.COLOR_BGR2HSV)
    mask = sc.hsv_rule(hsv, RULE_LOWER, RULE_UPPER)
    hit = int((mask & target_mask).sum())
    return {"selected": float(mask.mean()),
            "coverage": hit / int(target_mask.sum()),
            "precision": hit / int(mask.sum()) if mask.any() else 0.0,
            "mask": mask}
```

- One function, applied identically to seven images (PNG plus six JPEGs). If the scoring
  differed between conditions in any way, the comparison would be worthless — putting it in
  a function is how that is guaranteed rather than hoped for.
- `RULE_LOWER`, `RULE_UPPER` and `target_mask` are the *same objects* Experiment 2 built.
  Because the cells run in order, no constant is retyped and no drift is possible.
- The `bgr_to_rgb` before `cvtColor` is the same convention fix as Experiment 1 and 2.
- Returning the mask alongside the numbers lets the figure show the mask without recomputing
  it differently.

**Encoding in memory, not through files:**

```python
ok, buffer = cv2.imencode(".jpg", sc.bgr_to_rgb(reference),
                          [int(cv2.IMWRITE_JPEG_QUALITY), quality])
assert ok, f"JPEG encode failed at quality {quality}"
decoded = sc.bgr_to_rgb(cv2.imdecode(buffer, cv2.IMREAD_COLOR))
```

- `cv2.imencode` runs the full JPEG encoder and hands back the **compressed bytes** as a
  NumPy array, and `imdecode` reads them back. No temporary files, so nothing to clean up
  and nothing that could accidentally end up in `outputs/` — which matters because the spec
  says exactly **one** file may remain under `outputs/extension/`.
- `buffer.nbytes` is then the true compressed size, so the size/quality trade-off is
  measured rather than estimated.
- **`bgr_to_rgb` appears twice, on purpose.** OpenCV encodes what it believes is BGR and
  decodes to BGR, while our arrays are RGB. Reversing on the way in and again on the way out
  keeps the round trip honest; skipping *both* would also work by symmetry, but skipping
  one would silently swap red and blue and hand the rule a blue target — Experiment 1's
  failure, sneaking in through a side door.
- `assert ok` — `imencode` returns a success flag rather than raising. Ignoring it is how
  an unwritable format fails silently.

**The metrics per condition:**

```python
"mask_agreement_vs_png": round(float((scores["mask"] == base["mask"]).mean()), 5),
"max_rgb_error": int(np.abs(decoded.astype(np.int16) - reference.astype(np.int16)).max()),
```

- **Mask agreement** is the metric that answers the question most directly: what fraction of
  *all* pixels get the same decision (selected or not) as they did on the lossless
  reference. Coverage and precision describe the target; agreement describes the whole
  decision surface.
- **Max RGB error** is the counterweight. It measures how badly the *pixels* were damaged,
  independent of the rule. Keeping both is what allows the conclusion to distinguish "the
  compression was gentle" from "the compression was brutal but the rule did not care" —
  and the numbers below show it is the second.
- `.astype(np.int16)` before subtracting, for the unsigned-wrap reason established in
  Task 4.

**Reporting, carefully:**

```python
most_compressed = jpeg_rows.iloc[-1]                              # positional: the last row
least_coverage = jpeg_rows.loc[jpeg_rows["roi_coverage"].idxmin()] # label-based: the worst row
```

`iloc` indexes by position, `loc` by label — and after filtering `comparison` down to the
JPEG rows, the labels are 1…6, not 0…5. Mixing them up is a classic pandas slip that
produces a real row for the wrong condition and no error at all. "Most compressed" and
"worst coverage" are also genuinely different rows here (q10 and q60), so the two are
reported separately rather than conflated.

**The figure** is the single saved artifact: two rows of ROI thumbnails and masks for
PNG/q95/q40/q10, and a full-width chart of coverage, precision, and mask agreement against
quality, with the PNG reference and both Experiment 2 conditions drawn as horizontal
reference lines. `invert_xaxis()` puts high quality on the left so the x-axis reads as
"increasing compression," and each point is annotated with its file size.

---

## The measured result

| encoding | quality | size | selected fraction | ROI coverage | precision | mask agreement vs PNG | max RGB error |
|---|---|---|---|---|---|---|---|
| PNG (lossless) | — | **4.2 KiB** | 0.11015 | 0.92154 | 1.00000 | 1.00000 | 0 |
| JPEG | 95 | 24.1 KiB | 0.11272 | 0.94301 | 1.00000 | 0.99743 | 104 |
| JPEG | 80 | 15.8 KiB | 0.11274 | 0.94317 | 1.00000 | 0.99741 | 124 |
| JPEG | 60 | 12.5 KiB | 0.11225 | 0.93899 | 0.99990 | 0.99790 | 132 |
| JPEG | 40 | 11.0 KiB | 0.11258 | 0.94180 | 0.99997 | 0.99757 | 160 |
| JPEG | 20 | 8.8 KiB | 0.11405 | 0.95415 | 0.99997 | 0.99610 | 198 |
| JPEG | 10 | 7.2 KiB | 0.11355 | 0.94964 | 0.99969 | 0.99659 | 225 |

**The answer to the question: it does not break.** Across a 3.3× range of file sizes and
quality factors from 95 down to 10, ROI coverage stays within **0.939–0.954**, precision
never drops below **0.9997**, and the mask agrees with the lossless reference on at least
**99.6%** of pixels. Set that against Experiment 2, where the same rule went from 0.922
coverage on the normal capture to **0.000** on the dim one. Lighting is a first-order threat
to a color rule; storage quality, on this image, is a rounding error.

**Three things worth saying that the headline hides:**

- **The pixels really were damaged.** Max RGB error climbs monotonically from 104 at q95 to
  225 at q10 — nearly the full 8-bit range. The rule survives not because the compression
  was gentle but because the damage is concentrated as ringing on hard edges, while the
  large flat interior of the target keeps its hue and saturation. A rule that depended on
  edges rather than on regions would not have survived at all.
- **Coverage goes slightly *up* under compression** (0.922 → up to 0.954), which is worth
  checking rather than celebrating. At q10, 862 ROI pixels are selected that the lossless
  reference rejected — and **all 862** were outline-stroke pixels in the PNG (saturation 0,
  value 30). JPEG's smoothing bleeds red across the black stroke, lifting those pixels over
  both thresholds. A metric improving is not evidence of a better image; it is evidence that
  the metric and the compression artifact interact.
- **PNG is the smallest file.** 4.2 KiB versus 24.1 KiB at q95 and 7.2 KiB even at q10.
  This is a flat, synthetic image with large uniform regions and hard edges: exactly the
  content PNG's lossless run-length-plus-filter scheme handles best and exactly the content
  JPEG's block transform handles worst, since it must spend bits encoding the ringing it
  introduces at every edge. "JPEG is smaller" is a fact about photographs, not about images.

---

## What the analysis prompt is asking for

Cell `06afeab5` wants the question **and** the conclusion in 80–130 words total. The
structure that fits: one sentence of question, one of method (one rule, one image, quality
as the only variable, PNG as control), two of result with numbers (the coverage band and
mask agreement against Experiment 2's 0.000), and one of limitation — a synthetic
flat-color image is the friendliest possible case for both PNG and a region-based color
rule, so the finding should not be generalized to photographs without re-running it.

---

## Two concepts examiners love to probe

1. **Damaged pixels versus damaged decisions.** A max error of 225 levels and a mask
   agreement of 0.996 are both true at q10. Being able to hold those two facts together —
   and to explain *where* on the image the error lives — is the actual content of this
   extension.
2. **Why PNG beats JPEG here.** Lossless entropy coding of large flat regions versus a
   block transform that has to encode its own ringing artifacts. The corollary is that
   compression benchmarks depend on the content, and a synthetic test image is not evidence
   about photographs.
