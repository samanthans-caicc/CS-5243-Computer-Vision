# A1 Experiment 3 — Code Walkthrough

_Study note (not part of the graded submission). Explains `transform_frame` and the
Experiment 3 notebook cell (`e6418d31`)._

---

## Key background: a video is a sampled array, and the file is not the samples

Two ideas carry this experiment.

**1. Time is an index, and the conversion factor is the frame rate.** A video is a stack of
`(H, W, 3)` arrays plus one number, `fps`, that says how the stack maps to seconds:

```
frame index = round(time_in_seconds × fps)
```

`moving_shapes.mp4` is 48 frames at 12 fps, so 4.0 seconds, and `t = 2.0 s` is frame 24.
The *frame count* (48) never appears in that formula — using it instead is the units error
Task 5 documents.

**2. An MP4 is not a container for your array; it is a lossy re-encoding of it.** Writing a
frame and reading it back does not return the same numbers. H.264 quantizes in a
transformed color space, so the round trip preserves the *structure* — resolution, frame
rate, frame count, the visible content — while changing individual sample values. Any
verification therefore has to check structure, and any claim of "identical" has to be
qualified with a measurement. That is why `verify_video_round_trip` compares metadata and
not pixels.

---

## `transform_frame`

```python
arr = _check_rgb(frame_rgb)
if arr.dtype != np.uint8:
    raise ValueError(f"expected a uint8 frame, got {arr.dtype}")
out = arr[:, ::-1, :].astype(np.float64)
out[:, :, 0] *= 0.65                                  # scale red only
return np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)
```

- **`_check_rgb` plus a dtype gate.** `a1_tools.write_mp4` rejects anything that is not
  `HxWx3 uint8`, so failing here — where the message names the function you called — beats
  failing later inside the writer.
- **`arr[:, ::-1, :]`** reverses the **column** axis: all rows, columns backwards, all
  channels. Same horizontal-flip idea as Task 2's `flip_horizontal`, written with the
  channel axis spelled out because a frame is always 3-D here.
- **`.astype(np.float64)` does double duty.** It escapes `uint8` before the multiply (a
  scale of 0.65 in `uint8` would truncate every value and lose precision), *and* it makes a
  copy — which is what allows the next line to write in place with `*=` without ever
  touching the caller's frame. Non-mutation is verified in the notebook run.
- **`out[:, :, 0] *= 0.65`** — channel 0 is red under RGB. Only that plane is scaled;
  green and blue pass through untouched. This is a *channel* operation composed with a
  *spatial* one, which is the vocabulary Task 2 set up.
- **`np.clip(np.rint(out), 0, 255).astype(np.uint8)`** — the same
  round → clip → cast discipline as Task 3 and Task 4. Scaling by 0.65 can only shrink
  values, so the clip is defensive rather than load-bearing here; the rounding is not.
  Without it, `149 × 0.65 = 96.85` truncates to **96** instead of rounding to **97**, and
  because truncation only ever moves values down, the whole red channel picks up a
  systematic downward bias on top of the intended scaling.

---

## The notebook cell logic (`e6418d31`)

**Probe first:**

```python
probe_in = a1_tools.probe_video(video_in)
# {'path': 'moving_shapes.mp4', 'width': 320, 'height': 240, 'fps': 12.0,
#  'frame_count': 48, 'duration_seconds': 4.0, 'codec': 'h264'}
```

Everything downstream is derived from this dict rather than hard-coded: `fps` drives the
sampling and the writer, `width`/`height`/`frame_count` become the expectations for the
read-back check. Hard-coding 12 and 48 would make the cell pass for the wrong reason if the
asset ever changed.

**Sampling by time, not by index:**

```python
SAMPLE_TIMES = [0.0, 1.0, 2.0, 3.0]
samples = a1_tools.sample_video_frames(video_in, SAMPLE_TIMES)
index = round(t * fps)
```

`sample_video_frames` internally does `round(time * fps)` and then streams the video once,
collecting only the wanted indices. The cell recomputes the same index purely so it can
*print* it in each panel title — the figure therefore carries its own evidence that the
conversion used the frame rate. The contact sheet's two rows show each sampled frame as
decoded (top) and after `transform_frame` (bottom); the mirrored burnt-in timestamp text is
the giveaway that the flip really happened.

**Transform every frame, then write:**

```python
transformed = [sc.transform_frame(frame) for frame in a1_tools.iter_video_frames(video_in)]
a1_tools.write_mp4(video_out, transformed, fps=fps)
```

- `iter_video_frames` is a **generator** — it yields frames one at a time instead of
  decoding the whole video into memory. At 48 frames of 320×240 that hardly matters; the
  habit matters.
- `write_mp4` is the bundled imageio-ffmpeg path the spec requires (libx264, `yuv420p`,
  `macro_block_size=16`). OpenCV's `VideoWriter` is explicitly not graded, and it is the
  usual source of "the file exists but nothing can play it" — a different codec on every
  machine.
- The writer re-validates every frame (`HxWx3 uint8`) and raises on an empty sequence, so a
  silent zero-frame file is impossible.

**Verify structurally, then quantify the loss:**

```python
read_back = a1_tools.verify_video_round_trip(
    video_out, expected_width=..., expected_height=..., expected_fps=fps, expected_frames=len(transformed))
# checks: {'resolution': True, 'fps': True, 'frame_count': True} -> valid: True

decoded_first = next(iter(a1_tools.iter_video_frames(video_out)))
delta = np.abs(decoded_first.astype(np.int16) - transformed[0].astype(np.int16))
```

- The three checks use tolerances, not equality: fps within 5% and frame count within ±1,
  because encoders are allowed to nudge both. Demanding exact equality would produce a
  failure that means nothing.
- The delta measurement is the honest footnote to "valid: True". `.astype(np.int16)` before
  subtracting, once again, so an unsigned difference cannot wrap (Task 4's lesson applied to
  video).

**The JSON artifact:**

```python
video_metadata = {
    "input":  {**probe_in,  "path": video_in.name},
    "output": {**probe_out, "path": video_out.name},
    "read_back": {"valid": bool(...), "checks": {name: bool(ok) for name, ok in ...}, "metadata": ...},
    "transform": {"function": "student_code.transform_frame", "operations": [...], "frames_written": ...},
}
video_json.write_text(json.dumps(video_metadata, indent=2, default=str), encoding="utf-8")
```

- The nested key structure is dictated by `A1/config/A1.yml`, which lists required dotted
  keys — `input.width`, `output.fps`, `read_back.checks.frame_count`, and so on. The
  validator walks those paths and errors on any that is missing, so the shape of this dict
  is a requirement, not a preference.
- **`bool(...)` around the check results** matters: they are `np.bool_`, which
  `json.dumps` refuses to serialize. Same class of problem as `.item()` in Task 1 — NumPy
  scalars need unwrapping at any boundary that leaves NumPy.
- `default=str` is the safety net for anything else NumPy-shaped that slips through (a
  `Path`, a `np.float32`), so the artifact cannot fail to write at the very end of a long
  cell.
- Recording input **and** output **and** read-back is the point: it lets a reader confirm
  the transform preserved every structural property, without taking the notebook's word for
  it.

---

## The measured result

```
input : width 320, height 240, fps 12.0, frame_count 48, duration 4.0 s, codec h264
output: width 320, height 240, fps 12.0, frame_count 48, duration 4.0 s, codec h264
read-back checks: {'resolution': True, 'fps': True, 'frame_count': True} -> valid: True
frame 0 after H.264 re-encode: mean |difference| 0.86, max 46 levels
```

48 frames in, 48 frames out, same geometry and rate, written at 26.5 KiB. The round trip is
**structurally exact and numerically approximate**: on average each sample moves by less
than one level, but the worst sample moves by 46 — and those large errors sit on the hard
edges of the shapes, which is exactly where a block-transform codec spends the least of its
bit budget. It is the same phenomenon the Task 6 extension measures deliberately with JPEG.

---

## What the analysis prompt is asking for

"Cite input/output metadata and state what the sampling shows." The metadata is in
`video_metadata.json`; the sampling claim is that four timestamps one second apart map to
frames 0/12/24/36 and show the shapes in four distinct positions — i.e. the index/time
relationship is `fps`, and it is verifiable rather than assumed. The 0.86/46 delta is the
right place to note the limitation: the saved video is not a lossless copy of the array
that was written.

---

## Two concepts examiners love to probe

1. **`time × fps`, not `time × frame_count`.** Be ready to give the units argument: fps is
   frames *per second*, so seconds × (frames/second) = frames. Frame count is frames *per
   video*; multiplying by seconds gives frame-seconds, which indexes nothing. Task 5 shows
   what it does in practice.
2. **Lossy round trip.** "Verified" means the structure survived — resolution, rate, count.
   It does not mean the pixels are identical, and you should be able to quote the mean and
   max difference that prove they are not.
