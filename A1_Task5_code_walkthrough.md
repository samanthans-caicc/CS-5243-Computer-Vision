# A1 Task 5 — Code Walkthrough

_Study note (not part of the graded submission). Explains the failure-analysis notebook
cell (`4aac6891`) — one image-representation failure and one video/encoding failure._

---

## Key background: what makes something a failure *analysis*

Task 5 adds no new functions. It is graded on the shape of the argument, and the rubric
names four parts: **evidence → mechanism → correction → verification**. Every part of the
cell exists to supply one of them:

| part | what it means here |
|---|---|
| evidence | show the wrong output, with numbers, not a description of it |
| mechanism | say *why* it is wrong in terms of the representation — units, axis order, dtype |
| correction | the fixed call, run side by side under identical conditions |
| verification | a check that the correction actually did what you claim |

The two failures are deliberately chosen so that neither overlaps the ones already covered:
Task 4 owns dtype/range, Experiment 1 owns channel order. Task 5 takes **axis order** and
**time-to-index units**. Both share a property worth naming in the prose: they produce no
exception. Nothing crashes; you simply get the wrong pixels.

---

## Failure A — a region written `(x, y)` fed into `(row, column)` indexing

```python
BAR_XYWH = (243, 318, 190, 45)       # the yellow bar, measured from the image
TRIANGLE_XYWH = (506, 79, 119, 221)  # the blue triangle, near the right edge
bx, by, bw, bh = BAR_XYWH
wrong_crop = sc.crop_image(scene, top=bx, left=by, height=bh, width=bw)   # x and y swapped
right_crop = sc.crop_image(scene, top=by, left=bx, height=bh, width=bw)
```

**The setup is the argument.** `BAR_XYWH` is written the way image regions are written
almost everywhere outside NumPy — x first. `crop_image` takes `top` and `left`, i.e. row
first. The two orders look interchangeable at a call site, and nothing in the type system
objects: both are ints, both are in range.

- `crop_image(top=243, left=318, ...)` slices rows 243–288, columns 318–508. That is a
  perfectly valid rectangle. It is just the wrong one — it lands on the bottom edge of the
  green circle.
- `crop_image(top=318, left=243, ...)` slices rows 318–363, columns 243–433: the yellow bar.

**The evidence is a measurement, not a look:**

```python
wrong_mean = wrong_crop.reshape(-1, 3).mean(axis=0)   # (156.1, 199.9, 179.7)
right_mean = right_crop.reshape(-1, 3).mean(axis=0)   # (245.0, 210.0, 35.0)
```

`reshape(-1, 3)` flattens the rectangle to a list of pixels while keeping the channel axis,
so `mean(axis=0)` is the average RGB of the region. `(245, 210, 35)` is unmistakably yellow;
`(156, 200, 180)` is the pale-green/background mixture that the wrong rectangle actually
covers. Two numbers settle what two similar-looking crops cannot.

**The second case is the important one:**

```python
try:
    sc.crop_image(scene, top=tx, left=ty, height=th, width=tw)
    print("blue triangle with x and y swapped: no error (unexpected)")
except ValueError as error:
    print(f"blue triangle with x and y swapped: ValueError — {error}")
# -> ValueError: rectangle (top=506, left=79, h=221, w=119) is out of bounds for image (400, 640)
```

The blue triangle sits at x = 506 in a 640-wide, **400-tall** image. Swap the axes and the
row origin becomes 506, past the bottom of the image, so `_check_rectangle` (Task 2) raises.

That contrast is the point of including it: **the same mistake is loud or silent purely as
a function of geometry.** A region in the left portion of a landscape image swaps into a
valid rectangle and fails silently; a region near the right edge swaps out of bounds and
raises. You cannot rely on the exception to find this bug, which is why the mechanism
sentence matters more than usual.

**The figure** draws both rectangles on the source — green solid for `[row, col]`, red
dashed for `[x, y]` — then shows the two crops. Note the patch coordinates:
`plt.Rectangle((bx, by), ...)` for the correct box and `plt.Rectangle((by, bx), ...)` for
the wrong one, because matplotlib's `Rectangle` takes `(x, y)` while the crop takes
`(row, col)`. The figure code has to perform the very translation the failure is about.

---

## Failure B — a timestamp converted to a frame index with the frame count

```python
frames = list(a1_tools.iter_video_frames(video_in))
frame_count, fps = len(frames), float(probe_in["fps"])     # 48 frames, 12.0 fps
for t in TIMES:
    wrong_index = int(round(t * frame_count))   # frames per VIDEO, not frames per second
    right_index = int(round(t * fps))           # frames per SECOND -> a frame index
    clamped = min(wrong_index, frame_count - 1) # what an unchecked read actually returns
```

**The mechanism is a units error**, and it is worth writing out dimensionally:

- `fps` has units frames / second. `seconds × frames/second = frames`. ✔
- `frame_count` has units frames / video. `seconds × frames/video = frame·seconds/video`,
  which is not an index of anything. ✘

The two agree at `t = 0` — which is exactly why this survives a quick sanity check — and
then diverge by a factor of `frame_count / fps`, i.e. the **duration** (4 here). Every
non-zero timestamp overshoots the end of the video.

**Why it produces a plausible picture instead of an error:** the read is clamped, not
checked. `min(wrong_index, frame_count - 1)` is what a slice or a library seek does in
practice, so instead of `IndexError` you get frame 47 — a real frame from the real video —
for *every* timestamp. The failure disguises itself as "the video stopped moving."

**The table makes the divergence explicit:**

```python
rows.append({"time_s": t, "wrong_index (t * frame_count)": wrong_index, "clamped_to": clamped,
             "right_index (t * fps)": right_index, "in_range": wrong_index < frame_count,
             "identical_frame": bool(np.array_equal(frames[clamped], frames[right_index]))})
```

| t | wrong index | clamped to | right index | in range |
|---|---|---|---|---|
| 0 s | 0 | 0 | 0 | yes |
| 1 s | 48 | 47 | 12 | no |
| 2 s | 96 | 47 | 24 | no |
| 3 s | 144 | 47 | 36 | no |

`identical_frame` is the column that proves the failure is not cosmetic: it is `True` only
at `t = 0`, where the two formulas agree by coincidence.

**The verification** closes the loop:

```python
corrected = [frames[int(round(t * fps))] for t in TIMES]
distinct = all(not np.array_equal(a, b) for a, b in zip(corrected, corrected[1:]))
matches_helper = all(np.array_equal(a, b) for a, b in
                     zip(corrected, a1_tools.sample_video_frames(video_in, TIMES)))
```

- `distinct` — consecutive corrected frames are pairwise different, i.e. time actually
  advances. A correction that returned four copies of frame 0 would also "not crash."
- `matches_helper` — the corrected indices reproduce what the provided
  `sample_video_frames` returns, which is an independent implementation of the same rule.
  Agreeing with a second implementation is stronger than agreeing with yourself.

Both print `True`.

---

## The measured result

```
image shape (400, 640, 3) = (height, width, channels); the yellow bar is at (x, y, w, h) = (243, 318, 190, 45)
crop_image(top=243, left=318) — x fed into the row axis -> mean (R, G, B) = (156.1, 199.9, 179.7)
crop_image(top=318, left=243) — rows then columns       -> mean (R, G, B) = (245.0, 210.0, 35.0)
blue triangle with x and y swapped: ValueError — rectangle (top=506, left=79, h=221, w=119) is out of bounds for image (400, 640)

moving_shapes.mp4: 48 frames at 12 fps = 4.0 s of video
corrected frames are pairwise distinct: True; they match a1_tools.sample_video_frames: True
```

Figures: `failure_image.png` (source with both boxes, wrong crop, corrected crop) and
`failure_video.png` (four wrong frames over four correct ones).

---

## What the analysis prompt is asking for

The analysis cell (`098cf7e6`) is headed "Image failure analysis," but Section 7 asks for
both failures documented. For each, four sentences will do it if each one carries a number:
what was observed (the two mean-RGB triples; the repeated frame 47), the mechanism (row/col
vs x/y; frames-per-second vs frames-per-video), the correction (swap the arguments; use
`fps`), and the verification (the corrected crop's mean RGB; `distinct` and
`matches_helper` both `True`). The sentence most worth including is the one about silence:
neither bug raises in the case that matters, so both are caught by checking values, not by
waiting for an exception.

---

## Two concepts examiners love to probe

1. **Why the axis-order bug is dangerous rather than annoying.** It returns a valid,
   in-bounds array of the right shape and dtype. The only way to catch it is to check
   content — which is why the walkthrough measures a mean RGB instead of saying "it looks
   wrong."
2. **Dimensional analysis as a debugging tool.** `t × fps` versus `t × frame_count` can be
   settled without running anything, just by tracking units. Be ready to say why the two
   agree at `t = 0` and diverge by exactly the video's duration afterwards.
