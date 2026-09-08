# A1 — analysis drafts

Drafts for every `# TODO: WRITE ANALYSIS` cell in `A1/A1.ipynb`.
Every number below comes from the saved outputs of the last full run
(`outputs/tables/*.csv`, `video_metadata.json`, and the notebook's stored cell output).
Nothing here has been written into the notebook.

File kept at the repo root, outside `A1/`, so submission packaging never picks it up.

---

## Task 2 analysis (cell 10) — 2–4 sentences

The channel operations only reordered the last axis. `bgr_to_rgb` left the pixel at
(row 150, column 120) where it was but reversed its channels, so (235, 35, 35) became
(35, 35, 235), and `extract_channel(co, 0)` pulled out the red numbers unchanged, turning
(300, 720, 3) into a 2-D (300, 720). The spatial operations changed positions instead:
`flip_horizontal` reversed the columns, moving that same red pixel from column 120 to
column 599 with its value untouched, and `crop_image` kept only rows 30–269 and columns
30–269 to give (240, 240, 3). `replace_region` was the only one that changed values,
overwriting rows 0–29 of `wide_scene.jpg` with magenta (255, 0, 255).

---

## Task 3 analysis (cell 13) — 2–3 sentences

Over 5 repeats on the same image, the loop had a median of 232.94 ms and the vectorized
version 3.07 ms, so the array version was 75.9 times faster. The comparison is controlled
because both calls got the same image (`wide_scene.jpg`, 648,000 elements), the same offset
(+40), and the same rounding-and-clipping rule, so the only thing that changed was looping
in Python versus letting NumPy do the work in one pass. The outputs were equal, which is
what makes the timing a fair argument about strategy and not about two different answers.

---

## Task 4 analysis (cell 16) — 3–5 sentences

`to_float01` divides by the dtype maximum, not the observed maximum, so the scale belongs
to the representation and the round trip back through `to_uint8_safe` came back with a
maximum error of 0 levels. Adding 80 in uint8 wrapped 58.57% of the samples to a *darker*
value, because uint8 arithmetic goes modulo 256 before anything can clip it; the same
addition done in float and then clipped disagreed with it by as much as 255 levels. The
broken conversion fails a different way: after a gain of 1.4, 58.49% of the float samples
were above 1.0, and `(float_image * 255).astype(np.uint8)` truncates and wraps those instead
of saturating, disagreeing with `to_uint8_safe` on 62.89% of samples (max gap 255). The fix
is to clip to [0, 1] first, then scale by 255, then round — clipping before the cast is what
turns a wrap into a saturation. The uint16 ramp shows the same idea on the read side: a
plain `.astype(np.uint8)` is not monotonic across the middle row (the low byte sawtooths 173
times), while normalizing first and converting safely stays monotonic.

---

## Experiment 1 analysis (cell 19) — 4–6 sentences

All five panels use one source image and one fixed [0, 1] float display scale, so only the
interpretation changes. The channel-order error is the loud one in color: reading the same
bytes as BGR moves the red rectangle's mean from (209.7, 43.8, 43.8) to (43.8, 43.8, 209.7),
a mean absolute difference of 0.141 and a maximum of 0.824 on the [0, 1] scale. The
grayscale weighting choice is much smaller: BT.601 luminance minus the plain channel mean
has a mean signed difference of only 0.0041 and never differs by more than 0.146, and it
shows up as the red rectangle reading darker under luminance (0.366) than under the mean
(0.389), because red only carries a weight of 0.299. The evidence that separates the two is
that my luminance matches OpenCV's grayscale to within 0.0019 when OpenCV is called with the
BGR input it documents, so the weights agree and any larger gap is not a weighting question.
Calling OpenCV with the wrong convention is the dangerous case: 97.23% of pixels change, but
by at most 39 of 255 levels, so the result still looks like a believable grayscale image and
only a measurement gives it away.

---

## Experiment 2 analysis (cell 22) — 3–5 sentences

One rule (hue 170–179 and 0–10, saturation ≥ 100, value ≥ 60) was tuned on the normal image
and then applied unchanged, so the capture condition is the only variable. The warm capture
was the strongest condition: coverage of the shared target ROI rose to 100% because the cast
pushed the ROI median to (3, 234, 240), which sits comfortably inside every clause — but it
also selected more of the image (0.138 of pixels versus 0.110) and precision fell to 86.56%,
so the gain came with leakage. The dim capture was the weakest and failed completely, with
0% coverage and nothing selected at all. The clause pass rates say exactly why: inside the
ROI, hue still passed 100% and value still passed 92.15%, but saturation passed 0%, because
dimming dropped the ROI median saturation from 204 to 85, below the rule's lower bound of
100. This says something about this rule on these three aligned images, not about
segmentation in general.

---

## Experiment 3 analysis (cell 25) — 3–4 sentences

The input `moving_shapes.mp4` probes as 320×240, 12 fps, 48 frames, 4.0 s, h264, and the
output `a1_transformed.mp4` probes identically after all 48 frames were transformed and
written through the bundled ffmpeg path (26.5 KiB). `verify_video_round_trip` returned
resolution, fps, and frame_count all True, so what read-back proves is structural: the file
decodes, and it has the size, rate, and frame count I meant to write. What it does not prove
is that the samples survived — frame 0 came back from the H.264 re-encode with a mean
absolute difference of 0.86 and a maximum of 46 levels against the frame I handed the
writer. So a passing round-trip check means the container and timing are right, not that the
pixels are unchanged, which is also why frame indices have to come from fps rather than from
the frame count.

---

## Image failure analysis (cell 30, first prompt)

I wrote the yellow bar down the way regions usually get written, as (x, y, w, h) =
(243, 318, 190, 45), and fed x into the row argument. NumPy indexes (row, column) = (y, x),
so `crop_image(top=243, left=318)` returned a perfectly valid, in-bounds rectangle of the
wrong part of the image: mean (R, G, B) = (156.1, 199.9, 179.7) instead of the bar's
(245.0, 210.0, 35.0). Nothing raised, which is the real problem — the same swap on the blue
triangle at (506, 79) did raise `ValueError: rectangle (top=506, ...) is out of bounds for
image (400, 640)`, so whether this bug is loud or silent depends only on where the region
happens to sit. The correction is to pass `top=y, left=x`, and the corrected crop's mean
(245.0, 210.0, 35.0) is the yellow I expected, shown side by side in
`outputs/figures/failure_image.png`.

---

## Video failure analysis (cell 30, second prompt)

Converting a timestamp with `t * frame_count` instead of `t * fps` is a units error:
frame_count is frames per *video*, not frames per second, so the product is not a frame
index. On a 48-frame, 12 fps, 4.0 s clip it agrees only at t = 0 and then overshoots by
exactly the duration, 48 / 12 = 4: t = 1, 2, 3 s give indices 48, 96, and 144, all past the
last valid index 47. Because the read clamps instead of checking, every one of those came
back as frame 47, so the contact sheet showed the same final frame three times and the
shapes appeared frozen. Using `round(t * fps)` gives 12, 24, and 36; I verified the
correction two ways — the four corrected frames are pairwise distinct, and they match
`a1_tools.sample_video_frames` exactly. `outputs/figures/failure_video.png` shows both rows.

---

## Extension — question and conclusion (cell 33) — 80–130 words

**Question:** Experiment 2 showed this HSV rule is sensitive to the capture condition, and
storage is a capture condition too — how much JPEG compression can `color_shapes.png` survive
before the same rule stops selecting the same target?

**Conclusion:** It survives everything I tested. Against the lossless PNG reference
(4.2 KiB, coverage 0.9215, precision 1.0000), quality 10 still gives coverage 0.9496,
precision 0.9997, and 99.66% mask agreement, even though single samples move by up to 225
levels; coverage varies by only 0.0152 across all six qualities. JPEG damage lands on the
hard edges, not on the large flat target, so for this rule lighting matters far more than
compression — the dim capture scored 0.000. Worth noting: the PNG is also the smallest file.

---

## Final synthesis (cell 27) — at most 180 words

The main representation lesson is that the same bytes mean nothing until you say how to read
them. In `outputs/figures/color_representations.png`, reading the file as BGR instead of RGB
turns the red rectangle's mean (209.7, 43.8, 43.8) into (43.8, 43.8, 209.7), and the array
never complains.

The strongest experimental finding is in `experiment_metrics.csv`: one fixed HSV rule kept
92.15% of the target under normal light and 100% under the warm cast, but 0% when the image
was dimmed, and the clause pass rates pin the cause on saturation alone — the ROI median fell
from 204 to 85, below the rule's bound of 100, while hue still passed 100%.

The limitation I keep coming back to is that passing a check is not the same as being
correct. `video_metadata.json` reports resolution, fps, and frame count all True for
`a1_transformed.mp4`, yet frame 0 came back from the H.264 encode with a mean absolute
difference of 0.86 and a maximum of 46 levels. The structure round-trips; the pixel values
do not.

---

## Reflection (cell 35) — 100–150 words

The practice I am carrying forward is writing the check next to the result instead of
trusting that the code ran. Task 4 only became clear once I counted that 58.57% of samples
wrapped darker, and Failure A only looked like a bug once I compared crop means — both are
silent otherwise.

The result another person should reproduce exactly is Experiment 2: the same wrapped-hue rule
gives target coverage 0.9215 on `color_shapes.png`, 0.0000 on the dim version, and 1.0000 on
the warm version, since those depend only on the provided images and the fixed rule.

Two things are platform-dependent. The Task 3 timings are machine-specific — the 75.9×
speedup is the shape of the result, not a number to match. The H.264 round trip is also
encoder-dependent, so the frame-0 mean difference of 0.86 may vary with the ffmpeg build,
though it should stay small and nonzero.

**Assistance disclosure (Section 0) is already filled in as "Claude Code".**
