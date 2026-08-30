# CS 5243 Assignment 1 — Work Log / Cross-Device Handoff

_Last updated: 2026-08-30. This file is a scratch/handoff note, not part of the graded submission._

---

## ⚠️ MAJOR CHANGE (2026-08-30) — canonical layout is now `A1/`, old bundle deprecated

The repo was restructured to the **canonical course layout**: a top-level, git-tracked
**`A1/`** folder sits as a sibling of **`common-setup/`** (exactly what
`common-setup/README.md` describes). **This `A1/` is the operative assignment now.**

- The old **`Assignment-1/cs5243-A1/` bundle is deprecated** and will be **deleted by the
  user personally** (do not delete it yourself). Everything below that references
  `Assignment-1/...` paths, cell ids, or the status table is about the OLD bundle and is
  now **stale**.
- `A1/` is a **fresh starter**: `A1/src/student_code.py` has all **15 functions as
  `NotImplementedError` stubs again** (including `image_summary` — the old impl did NOT
  carry over). `A1/src/a1_tools.py` is a **new version** (differs from the old one).
- **Paths that changed:** notebook `A1/A1.ipynb` (new cell ids — old ids in this file are
  invalid) · dataset now `A1/data/images` + `A1/data/video` (was `assets/A1/`) · config
  `A1/config/A1.yml` · tests `A1/tests_public/` · validate/package now run from
  **`common-setup/scripts/`** targeting sibling `A1/`.
- **`cs5243` is now editable-installed from `common-setup/`** (the newer, canonical copy),
  not the old bundle. `import cs5243` → `common-setup/cs5243`.
- The **code drafts further down (image_summary, Task 2 functions) are still useful
  reference** — the 15 signatures are unchanged — but re-verify against the new
  `a1_tools.py` and paste into the NEW notebook's cells (find current ids in `A1/A1.ipynb`).
- Identity in the new notebook is still the **`REPLACE ME` placeholder** — the student must
  re-enter name + ABC123 (was Samantha Salas / xso947 in the old bundle).

**Setup status on this WSL device (verified 2026-08-30):** env ready and pointed at the
canonical layout. `verify_environment()` clean · public tests `A1/tests_public` = 3 pass /
3 placeholder-skip · `common-setup/scripts/validate_submission.py --assignment A1
--preflight` = `valid: true`, 0 errors.

---

## STATUS AT A GLANCE (⚠️ OLD BUNDLE — reset to zero under new `A1/`)

_The table below reflects the deprecated `Assignment-1/` bundle. Under the fresh `A1/`
starter, **all 15 functions are unimplemented stubs** and no outputs/analysis exist yet._

| Part | Points | State |
|---|---|---|
| Task 1 — image arrays | 8 | **DONE & committed** (impl + CSV + figure). Analysis prose drafted, minor fixes pending. |
| Task 2 — NumPy spatial/channel ops | 14 | **Code ready (below), NOT yet applied** to `student_code.py` or notebook. |
| Task 3 — loop vs vectorized brighten | 8 | Not started |
| Task 4 — dtype/range safety | 10 | Not started |
| Experiment 1 — channel order / grayscale | 10 | Not started |
| Experiment 2 — HSV rule stability | 9 | Not started |
| Experiment 3 — video sampling | 6 | Not started |
| Task 5 — failure analysis | 10 | Not started |
| Task 6 — extension | 10 | Not started |
| Synthesis / reflection / submission | 9 | Not started |

---

## ENVIRONMENT SETUP ON A NEW DEVICE (do this first)

> **This WSL device (verified 2026-08-30):** env is already built and ready for the
> canonical `A1/` layout. conda/mamba live at `~/miniforge3` (there was no conda
> originally — Miniforge was installed fresh). The env interpreter is
> `~/miniforge3/envs/cs5243/bin/python` (Python 3.11.16). **`cs5243` is editable-installed
> from `common-setup/`** (`pip install -e common-setup`) → `import cs5243` resolves to
> `common-setup/cs5243`. `verify_environment()` returns no problems; public tests
> (`A1/tests_public`) pass (3 pass, 3 skip = unimplemented placeholders). To use it
> without activating: prefix commands with `~/miniforge3/envs/cs5243/bin/python`.
> Launch the notebook: `~/miniforge3/envs/cs5243/bin/jupyter lab A1/A1.ipynb`.
> Steps 1–2 below are only needed on a *different* fresh device (there, install editable
> from `common-setup`, not the old bundle).

0. **If `conda`/`mamba` is not installed at all** (as was the case here), install
   Miniforge first (matches the `conda-forge` channel in `environment.yml`):
   ```bash
   curl -fsSL -o /tmp/miniforge.sh \
     "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
   bash /tmp/miniforge.sh -b -p ~/miniforge3
   ```

1. **Create the conda env** (Python 3.11) from the course spec:
   ```bash
   ~/miniforge3/bin/mamba env create -f Assignment-1/cs5243-A1/environment.yml
   ```
   The env is named `cs5243`. Note: `environment.py` requires Python **exactly
   3.11** AND importable numpy/scipy/pandas/matplotlib/PIL/cv2/skimage/sklearn/
   imageio/tqdm/torch/torchvision/yaml — so the full env (incl. pytorch) is
   required for the notebook's Section 2 to pass; a bare venv won't do.

2. **Editable-install the course package FROM THE CURRENT REPO PATH:**
   ```bash
   <path-to>/envs/cs5243/bin/pip install -e Assignment-1/cs5243-A1
   ```
   ⚠️ The original clone shipped with a **stale editable install** pointing at an old
   location (`/home/sqmi/CS 5243 Computer Vision Assignments/...`, with spaces) that no
   longer exists. If the notebook's first cell fails with `ModuleNotFoundError: No module
   named 'cs5243'`, or only works when the CWD happens to be `cs5243-A1/`, re-run the
   `pip install -e` above from wherever the repo now lives.

3. **Launch Jupyter from the `cs5243` env** so the kernel's `sys.executable` is
   `.../envs/cs5243/bin/python`. Verify in a cell: `import sys; print(sys.executable)`.

### Notebook run discipline
- In a fresh kernel, **always run the setup cell first** (id `88492427`, starts with
  `from pathlib import Path`). It defines `iio`, `np`, `pd`, `plt`, `a1_tools`, `sc`,
  `PATHS`. `NameError: name 'a1_tools' is not defined` == you skipped it (or restarted
  the kernel and didn't re-run it).
- **Kernel → Restart Kernel and Run All Cells** is the safe reset. It will stop at the
  first not-yet-implemented task's `NotImplementedError`, which is expected.

### Public tests
```bash
<env>/bin/python -m pytest Assignment-1/cs5243-A1/assignments/A1/tests_public/test_a1_public.py -q
```

---

## REPO / STRUCTURE NOTES

- `Assignment-1/` was **embedded directly** into this repo (commit `a540c80`). Its old
  nested `.git` and the separate `github.com/samanthans-caicc/Assignment-1` repo are no
  longer used from here.
- **Two `cs5243` packages exist — do not mix them.** The repo has both a top-level
  `common-setup/` (the newer *canonical* course infra the `common-setup/README.md`
  describes) AND the older, self-contained `Assignment-1/cs5243-A1/` bundle. They diverge
  in `cs5243/data.py`, `cs5243/validation.py`, and the two `scripts/`. The **Assignment-1
  bundle is the operative world for A1** — it holds the notebook, `config/A1.yml`, matching
  tests, and working validate/package scripts. `import cs5243` is (correctly) editable-
  installed from the Assignment-1 copy; its preflight validator runs clean
  (`valid: true`, 0 errors). The canonical `common-setup/` sibling-layout the README assumes
  was **never assembled** here (no top-level `A1/`, and `common-setup/config/` has no
  `A1.yml`), so `common-setup/`'s own scripts can't validate A1 — treat `common-setup/` as
  reference only. `environment.yml` is byte-identical in both, so the one `cs5243` conda env
  serves either. Always run validate/package from `Assignment-1/cs5243-A1/scripts/`.
- All assignment work happens under **`Assignment-1/cs5243-A1/assignments/A1/`**:
  - `A1.ipynb` — the notebook
  - `src/student_code.py` — the 15 graded functions (the only file you implement)
  - `src/a1_tools.py` — provided helpers; **do not edit**
  - `outputs/{figures,tables,videos,extension}/` — generated artifacts
  - `assets/A1/` — dataset; **do not move or edit**
  - `tests_public/test_a1_public.py` — sanity tests

### Dataset (from the Task 1 run)

| filename | shape | dtype | min | max | mean |
|---|---|---|---|---|---|
| channel_order.png | (300, 720, 3) | uint8 | 35 | 255 | 86.47 |
| color_shapes.png | (400, 640, 3) | uint8 | 30 | 245 | 179.68 |
| color_shapes_dim.png | (400, 640, 3) | uint8 | 16 | 121 | 100.23 |
| color_shapes_warm.png | (400, 640, 3) | uint8 | 7 | 255 | 165.47 |
| intensity_ramp_16bit.png | (256, 512) | uint16 | 0 | 65535 | 23065.51 |
| portrait_scene.png | (540, 360, 3) | uint8 | 35 | 240 | 190.93 |
| wide_scene.jpg | (300, 720, 3) | uint8 | 0 | 255 | 162.16 |

Video `assets/A1/video/moving_shapes.mp4`: 320×240, 48 frames, 12.0 fps, ~4 s.

Named-pixel readouts on `channel_order.png` (row, col → R,G,B):
`(150,120)→(235,35,35)` red block · `(150,360)→(35,210,65)` green block ·
`(150,600)→(40,90,240)` blue block. Confirms imageio returns **RGB** order.

---

## TASK 1 — DONE (commits `4b55639`, `6000717`)

- `image_summary` implemented in `student_code.py` (see reference below).
- Task 1 notebook cell (id `0456cb0b`) implemented; the leftover
  `raise NotImplementedError` is commented out.
- `outputs/tables/image_metadata.csv` and `outputs/figures/image_overview.png` generated
  and committed.
- **Analysis prose**: the draft lives in a *separate* markdown cell (id `ffa50a2f`)
  immediately after the `**Task 1 analysis ...** YOUR RESPONSE` placeholder (id
  `6bada4c0`). Current text:

  > Shape order given by (height, width, channels) compared to (width, height) boils down
  > to how the x and y values are interpreted. For instance, channel_order.png and
  > wide_scene.jpg have the shape (300, 720, 3). The form (height, width, channels) takes
  > on the simplified version of (y, x, [R, G, B]). In channel_order.png, an array value
  > is (235, 35, 35) at (row 150, column 120) representing the color of that specific
  > pixel which is red-dominant. This differs from the (width, height) (or (x, y)) shape
  > order since these numbers represents the size of an image (e.g. 1920 x 1080).

  **Pending fixes:**
  - grammar: "these numbers represents" → "these numbers represent"
  - (optional) name where `(width, height)` order is used: `PIL.Image.size`,
    `cv2.resize`, screen resolutions
  - (optional) add axis-direction note: row index 0 at top, increases downward; column
    index increases rightward
  - (optional) fold the text into cell `6bada4c0` and delete the `YOUR RESPONSE` line, or
    leave as-is if the validator doesn't care

### Reference — `image_summary` (current, working)

```python
def image_summary(image: np.ndarray) -> dict[str, object]:
    """Return shape, height, width, channels, dtype, range, mean, and nbytes."""
    arr = np.asarray(image)
    if arr.ndim == 2:                       # grayscale: no channel axis
        height, width = arr.shape
        channels = 1
    elif arr.ndim == 3:                     # color: (H, W, C) with C == 3 or 4
        height, width, channels = arr.shape
    else:
        raise ValueError(f"Malformed image array. Expected a 2-D or 3-D image array, got shape {arr.shape!r}")
    return {
        "shape": tuple(int(n) for n in arr.shape),
        "height": int(height),
        "width": int(width),
        "channels": int(channels),
        "dtype": str(arr.dtype),
        "range": (arr.min().item(), arr.max().item()),
        "mean": float(arr.mean()),
        "nbytes": int(arr.nbytes),
    }
```

### Reference — Task 1 notebook cell (id `0456cb0b`)

```python
# STUDENT WORK — Task 1
image_paths = a1_tools.discover_images()

# --- load each image preserving its native dtype (imageio keeps 8- vs 16-bit, returns RGB) ---
loaded, records = {}, []
for path in image_paths:
    array = iio.imread(path)
    loaded[path.name] = array
    s = sc.image_summary(array)
    low, high = s["range"]
    records.append({
        "filename": path.name, "shape": s["shape"], "height": s["height"],
        "width": s["width"], "channels": s["channels"], "dtype": s["dtype"],
        "minimum": low, "maximum": high, "mean": round(s["mean"], 4), "bytes": s["nbytes"],
    })

metadata = pd.DataFrame.from_records(records)
csv_path = PATHS["tables"] / "image_metadata.csv"
metadata.to_csv(csv_path, index=False)
print("wrote", csv_path)
display(metadata)

# --- three named pixels: a channel-order swatch across channel_order.png ---
co = loaded["channel_order.png"]
named_pixels = {
    "channel_order.png RED block  (row 150, col 120)": co[150, 120],
    "channel_order.png GREEN block (row 150, col 360)": co[150, 360],
    "channel_order.png BLUE block  (row 150, col 600)": co[150, 600],
}
for label, value in named_pixels.items():
    print(f"{label}: array value (R, G, B) = {tuple(int(c) for c in value[:3])}")

# --- captioned overview figure ---
cols = 4
rows = (len(image_paths) + cols - 1) // cols
fig, axes = a1_tools.axes_grid(rows, cols)
for ax in axes.flat:
    ax.axis("off")
for ax, path in zip(axes.flat, image_paths):
    a = loaded[path.name]
    if a.ndim == 2:  # grayscale / uint16: scale to [0, 1] for display only
        ax.imshow(a, cmap="gray", vmin=0, vmax=np.iinfo(a.dtype).max)
    else:
        ax.imshow(a[..., :3])
    ax.set_title(f"{path.name}\n{a.shape} / {a.dtype}", fontsize=8)
fig.suptitle("A1 image set - native (height, width[, channels]) and dtype per file; "
             "grayscale/uint16 shown scaled for display only.", fontsize=10)
fig.tight_layout()
overview_path = PATHS["figures"] / "image_overview.png"
fig.savefig(overview_path, dpi=150, bbox_inches="tight")
print("wrote", overview_path)
plt.show()
```

---

## TASK 2 — CODE READY, NOT YET APPLIED (14 pts)

**Spec:** implement crop, horizontal flip, channel extraction, channel reorder,
rectangular replacement, contact-sheet construction. Pure NumPy — **no OpenCV** functions
that directly perform these ops. Crops and replacements must be **positive, in bounds, and
non-mutating**. `contact_sheet` takes **all-grayscale OR all-RGB `uint8`** images,
top-left-aligned, with an **8-pixel outer border and gutter**. Demo with
`channel_order.png`, `portrait_scene.png`, `wide_scene.jpg` and save
`outputs/figures/numpy_manipulations.png` with a caption. Analysis cell `a0e6beb4`:
"identify what was reordered vs what changed spatially."

**Verified in scratch** (`/tmp` scratchpad): public-test parity, non-mutation,
negative/zero-size/out-of-bounds rejection, contact-sheet border/gutter math, grayscale +
RGB, mixed-type rejection — all pass.

### `student_code.py` — add this helper above `crop_image`

```python
def _check_rectangle(image: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    """Raise ValueError unless (top, left, height, width) is a positive, in-bounds rectangle."""
    if min(int(top), int(left)) < 0 or min(int(height), int(width)) <= 0:
        raise ValueError(f"rectangle needs origin >= 0 and positive size, got "
                         f"top={top}, left={left}, height={height}, width={width}")
    if top + height > image.shape[0] or left + width > image.shape[1]:
        raise ValueError(f"rectangle (top={top}, left={left}, h={height}, w={width}) "
                         f"is out of bounds for image {image.shape[:2]}")
```

### `student_code.py` — replace the six stubs

```python
def crop_image(image: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    """Return an independent NumPy crop; reject nonpositive or out-of-bounds rectangles."""
    arr = np.asarray(image)
    _check_rectangle(arr, top, left, height, width)
    return arr[top:top + height, left:left + width].copy()


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    """Return a horizontally flipped view or copy using array operations."""
    return np.asarray(image)[:, ::-1].copy()


def extract_channel(image: np.ndarray, channel: int) -> np.ndarray:
    """Return one 2-D channel without changing its values."""
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D (H, W, C) image, got shape {arr.shape}")
    if not 0 <= channel < arr.shape[2]:
        raise ValueError(f"channel {channel} out of range for {arr.shape[2]} channels")
    return arr[:, :, channel].copy()


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Reorder a three-channel BGR array to RGB using NumPy."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected a 3-channel image, got shape {arr.shape}")
    return arr[:, :, ::-1].copy()


def replace_region(image: np.ndarray, top: int, left: int, height: int, width: int,
                   value: int | Sequence[int]) -> np.ndarray:
    """Return a copy with a positive, in-bounds rectangle replaced."""
    arr = np.asarray(image)
    _check_rectangle(arr, top, left, height, width)
    out = arr.copy()
    out[top:top + height, left:left + width] = value
    return out


def contact_sheet(images: Sequence[np.ndarray], columns: int, fill_value: int = 0) -> np.ndarray:
    """Top-left-align all-grayscale or all-RGB uint8 images with an 8-pixel border/gutter."""
    items = [np.asarray(im) for im in images]
    if not items:
        raise ValueError("contact_sheet needs at least one image")
    if int(columns) <= 0:
        raise ValueError(f"columns must be positive, got {columns}")

    is_rgb = items[0].ndim == 3
    for im in items:
        if im.dtype != np.uint8:
            raise ValueError(f"all images must be uint8, got {im.dtype}")
        ok = (im.ndim == 3 and im.shape[2] == 3) if is_rgb else (im.ndim == 2)
        if not ok:
            raise ValueError("contact_sheet requires all-grayscale OR all-RGB uint8 images")

    border = gutter = 8
    cell_h = max(im.shape[0] for im in items)
    cell_w = max(im.shape[1] for im in items)
    rows = (len(items) + columns - 1) // columns
    height = 2 * border + rows * cell_h + (rows - 1) * gutter
    width = 2 * border + columns * cell_w + (columns - 1) * gutter
    shape = (height, width, 3) if is_rgb else (height, width)
    sheet = np.full(shape, fill_value, dtype=np.uint8)

    for idx, im in enumerate(items):
        r, c = divmod(idx, columns)
        y0 = border + r * (cell_h + gutter)
        x0 = border + c * (cell_w + gutter)
        sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return sheet
```

### Task 2 notebook cell (id `c351ab6d`)

```python
# STUDENT WORK — Task 2
paths = {p.name: p for p in a1_tools.discover_images()}
co       = iio.imread(paths["channel_order.png"])
portrait = iio.imread(paths["portrait_scene.png"])
wide     = iio.imread(paths["wide_scene.jpg"])

# --- individual operations ---
crop      = sc.crop_image(co, top=30, left=30, height=240, width=240)   # spatial: the RED block
red_plane = sc.extract_channel(co, 0)                                   # channel: 2-D red plane
swapped   = sc.bgr_to_rgb(co)                                           # channel: reverse last axis
flipped   = sc.flip_horizontal(co)                                      # spatial: reverse columns
banner    = sc.replace_region(wide, 0, 0, 30, wide.shape[1], (255, 0, 255))  # overwrite a rectangle

# --- non-mutation check ---
before = iio.imread(paths["channel_order.png"])
for _ in (sc.crop_image(co, 0, 0, 10, 10), sc.flip_horizontal(co),
          sc.bgr_to_rgb(co), sc.replace_region(co, 0, 0, 10, 10, 0)):
    pass
assert np.array_equal(co, before), "a Task 2 op mutated its input"

# --- one coherent comparison across all three images ---
gallery = sc.contact_sheet(
    [co, portrait, wide,
     sc.flip_horizontal(co), sc.flip_horizontal(portrait), sc.flip_horizontal(wide)],
    columns=3, fill_value=30,
)

fig, axes = a1_tools.axes_grid(2, 3)
for ax in axes.flat:
    ax.axis("off")
axes[0, 0].imshow(crop);      axes[0, 0].set_title(f"crop_image {crop.shape[:2]}\nspatial: red block", fontsize=8)
axes[0, 1].imshow(red_plane, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("extract_channel(co, 0)\nred plane, 2-D", fontsize=8)
axes[0, 2].imshow(swapped);   axes[0, 2].set_title("bgr_to_rgb(co)\nchannel: R<->B, pixels fixed", fontsize=8)
axes[1, 0].imshow(banner);    axes[1, 0].set_title("replace_region(wide, ...)\nrectangle overwritten", fontsize=8)
axes[1, 1].imshow(flipped);   axes[1, 1].set_title("flip_horizontal(co)\nspatial: columns reversed", fontsize=8)
axes[1, 2].imshow(gallery);   axes[1, 2].set_title("contact_sheet(cols=3)\n8-px border + gutter", fontsize=8)
fig.suptitle("Task 2 - spatial ops (crop / flip / replace change WHERE pixels are) vs "
             "channel ops (extract / bgr_to_rgb reindex the last axis; positions unchanged). "
             "Comparison spans channel_order.png, portrait_scene.png, wide_scene.jpg.", fontsize=9)
fig.tight_layout()
out = PATHS["figures"] / "numpy_manipulations.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
plt.show()
```

**After applying:** run setup cell → Task 1 cell → Task 2 cell. Then write the Task 2
analysis (cell `a0e6beb4`): flip/crop/replace change *where* pixels sit (spatial);
`extract_channel`/`bgr_to_rgb` only reindex the last axis so colors change but positions
do not.

---

## REMAINING WORK (not started) — function + artifact checklist

All functions live in `src/student_code.py`. Cell ids are the code cell for each task.

### Task 3 — `brighten_loop`, `brighten_vectorized` (8 pts) · cell `d58039dc` · analysis `a0c07d3a`
- Both add `offset`, then `np.rint` (round-half-to-even) **before** clipping to `[0, 255]`
  and casting to `uint8`. Loop version uses explicit Python loops; vectorized uses array
  ops. They must produce identical output.
- Time each ≥3× on the same image with `time.perf_counter`; report **median ms**, repeat
  count, and output equality → `outputs/tables/timing.csv`.
- Public test: `brighten_*(np.array([[0,10,250]], uint8), 20) == [[20,30,255]]`.

### Task 4 — `to_float01`, `to_uint8_safe` (10 pts) · cell `39aeb486` · analysis `6d7db7ff`
- `to_float01`: `uint8`/`uint16` → `float32` in `[0, 1]` (divide by the dtype max).
- `to_uint8_safe`: `float [0,1]` → clip to `[0,1]`, round, ×255, `uint8`.
- Diagnose/repair `wrong = (float_image * 255).astype(np.uint8)` (no clip → wrap).
- Compare uint8 addition vs float addition; include the uint16 ramp →
  `outputs/figures/dtype_range_experiment.png`. ≥3 quantitative comparisons →
  `outputs/tables/experiment_metrics.csv` (this CSV is cumulative — Exp 2 appends to it).
- Public test: `to_uint8_safe(to_float01(uint8_arr)) == uint8_arr`, and
  `to_float01(...).dtype == float32`.

### Experiment 1 — `grayscale_mean`, `grayscale_luminance` (10 pts) · cell `be37e794` · analysis `3349cb7a`
- `grayscale_mean`: unweighted channel mean, `float32`.
- `grayscale_luminance`: `0.299R + 0.587G + 0.114B`, `float32`.
- Compare: correct RGB display · BGR-shown-as-RGB · channel mean · luminance · OpenCV
  grayscale (with correct input convention). Fixed source image and display limits →
  `outputs/figures/color_representations.png`.

### Experiment 2 — `hsv_rule` (9 pts) · cell `c5372f53` · analysis `baee3dde`
- Boolean mask for **inclusive** OpenCV-HSV bounds, **including wrapped hue** intervals
  (when `lower[0] > upper[0]`, hue wraps around 180).
- One rule applied to `color_shapes.png` / `_dim` / `_warm` (aligned). Report
  selected-pixel fraction + one more metric per condition; **append** to
  `experiment_metrics.csv`. Figure → `outputs/figures/hsv_stability.png`.

### Experiment 3 — `transform_frame` (6 pts) · cell `e6418d31` · analysis `64773af5`
- `transform_frame`: flip RGB frame horizontally, multiply red channel by `0.65`, return
  `HxWx3 uint8`.
- `a1_tools.probe_video(...)` → record input/output/read-back structure in
  `outputs/tables/video_metadata.json`.
- Extract frames near 0.0/1.0/2.0/3.0 s → `outputs/figures/video_contact_sheet.png`.
- Apply `transform_frame` to all frames, write `outputs/videos/a1_transformed.mp4` via
  `a1_tools.write_mp4`, verify with `a1_tools.verify_video_round_trip`.

### Task 5 — failure analysis (10 pts) · cell `4aac6891` · analysis `098cf7e6`
- Two failures: one image-representation, one video/encoding. Suggested video case:
  frame index = `time * frame_count` (wrong) vs `time * FPS` (right). Show wrong result,
  explain the units error, correct + verify. Figures → `outputs/figures/failure_image.png`,
  `outputs/figures/failure_video.png`.

### Task 6 — extension (10 pts) · cell `f390eecc` · answer `06afeab5`
- One question, one controlled comparison, one principal saved output, one conclusion
  (80–130 words). Keep exactly one file in `outputs/extension/` (png/jpg/mp4/csv/json/txt).
  No new packages.

### Wrap-up
- Section 6 synthesis: cell `92d7684e` (≤180 words).
- Section 9 reflection: cell `a6e760e3` (100–150 words). Section 0 disclosure already
  lists "Claude Code".
- Submission: Restart & Run All → export `A1.html` →
  `python <repo>/Assignment-1/cs5243-A1/scripts/validate_submission.py --assignment A1` →
  `python <repo>/Assignment-1/cs5243-A1/scripts/package_submission.py --assignment A1` →
  submit `Salas_Samantha_A1.zip`.

---

## TO CONTINUE ON ANOTHER DEVICE

```bash
git pull
# recreate env + editable install (see ENVIRONMENT SETUP above)
# open Assignment-1/cs5243-A1/assignments/A1/A1.ipynb, run the setup cell,
# then paste the Task 2 code from this file into student_code.py and cell c351ab6d.
```
