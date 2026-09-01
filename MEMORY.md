# CS 5243 Assignment 1 — Work Log / Cross-Device Handoff

_Last updated: 2026-09-01. This file is a scratch/handoff note, not part of the graded submission._

---

## ✅ PROGRESS (2026-09-01, later session) — ALL CODE DONE; only prose + submission left

**Every remaining code cell is implemented and the whole notebook runs clean.** Headless
`nbconvert --execute --allow-errors` over `A1/A1.ipynb`: **0 errored cells** (previously the
run stopped at Task 4's `NotImplementedError`). Public tests: **6 passed, 0 skipped**.

- **All 15 functions in `A1/src/student_code.py` are implemented.** New this session:
  `to_float01`, `to_uint8_safe`, `grayscale_mean`, `grayscale_luminance`, `hsv_rule`,
  `transform_frame`, plus a shared `_check_rgb` guard. `to_float01` divides by the *dtype*
  maximum (not the observed max); `to_uint8_safe` clips **before** scaling and rounds in
  float64 so `k/255 -> k` round-trips exactly; `hsv_rule` handles the wrapped-hue case
  (`lower[0] > upper[0]` -> `hue >= low | hue <= high`).
- **Task 4** (cell `39aeb486`): round trip (max error 0), uint8 `+80` wraps **58.6%** of
  samples darker, `(float*255).astype(uint8)` after a ×1.4 gain disagrees with
  `to_uint8_safe` on **62.9%** of samples (max gap 255), and the uint16 ramp cast straight to
  uint8 sawtooths **173 times** across its middle row. Figure + 7 metric rows.
  This cell also defines **`record_metrics(experiment, rows)`**, which rewrites only its own
  experiment's rows in `experiment_metrics.csv` — so re-running any cell never duplicates.
- **Experiment 1** (`be37e794`): five representations of `color_shapes.png`, all float32 with
  display limits pinned to [0, 1] (the sixth panel is a stretched difference *diagnostic*,
  labeled as such). OpenCV grayscale is called with BGR input; the wrong-convention call is
  kept as a measured control.
- **Experiment 2** (`c5372f53`): one rule — hue 170→10 wrapped, S ≥ 100, V ≥ 60 — on the three
  aligned images. **normal**: 11.02% selected, coverage 92.2%, precision 100%. **dim**: 0%
  everything — the target's saturation falls to 85, under the rule's 100 (the value clause
  still passes at 92.2%, so the failure is attributable to saturation alone). **warm**:
  coverage 100% but precision 86.6% — the warm cast pulls other shapes' outlines into the
  hue window.
- **Experiment 3** (`e6418d31`): 48 frames transformed and written with `a1_tools.write_mp4`;
  read-back valid (resolution/fps/frame_count all True). Frame 0 after H.264 re-encode differs
  by mean 0.86 / max 46 levels — structure round-trips, exact samples do not.
- **Task 5** (`4aac6891`): (A) a region written `(x, y, w, h)` fed into `(row, col)` indexing —
  the yellow bar `(243, 318, 190, 45)` silently returns the bottom of the green circle
  (mean RGB 156,200,180 vs the correct 245,210,35); the same swap on the blue triangle raises
  `ValueError`, which is the point: the bug is loud or silent purely by geometry.
  (B) `t * frame_count` vs `t * fps` — indices 48/96/144 clamp to frame 47 for every timestamp.
- **Task 6** (`f390eecc`): question = does JPEG compression break the Experiment 2 rule?
  Six quality levels vs the lossless PNG. Coverage stays 0.939–0.954 and mask agreement
  ≥ 0.996 all the way down to q10, against **0.000 for the dim capture** — storage is a far
  weaker threat to the rule than lighting. Side finding: on this flat synthetic image the PNG
  (4.2 KiB) is *smaller* than every JPEG (24.1 KiB at q95, 7.2 KiB at q10).
  One file in `outputs/extension/`: `jpeg_quality_vs_hsv_rule.png`.

**Bug fixed in already-"done" work:** `timing.csv` was written with columns
`implementation` / `outputs_identical`, but `A1/config/A1.yml` requires `method`,
`median_ms`, `repeats`, `outputs_equal`. The validator's `artifact-schema` check would have
**errored** on it. Cell `d58039dc` now writes the schema names (extra columns are allowed).

**Validator state now** (`python common-setup/scripts/validate_submission.py --assignment A1`):
2 error *kinds* left, both expected — `not-executed A1.ipynb` and `stale-html A1.html`. Every
artifact-schema check passes (`timing.csv`, `image_metadata.csv`, `experiment_metrics.csv`,
`video_metadata.json`). Both clear once the notebook is genuinely Restart-and-Run-All'd and
the HTML is re-exported.

**Housekeeping done:** the 8 tracked `.pyc` files are now `git rm --cached`'d (index only —
files left on disk, nothing committed).

### 📓 STUDY AIDS — one code walkthrough per task/experiment (2026-09-01)

Nine plain-markdown walkthroughs now sit at the repo root, all in the same house style
(key background → function-by-function code walk → notebook cell logic → measured result →
"two concepts examiners love to probe"):

| file | covers |
|---|---|
| `A1_Task1_code_walkthrough.md` | `image_summary`, cell `0456cb0b`, the metadata table |
| `A1_Task2_code_walkthrough.md` | the six spatial/channel ops, views vs copies |
| `A1_Task3_code_walkthrough.md` | `brighten_loop`/`brighten_vectorized`, timing harness |
| `A1_Task4_code_walkthrough.md` | `to_float01`, `to_uint8_safe`, `record_metrics`, cell `39aeb486` |
| `A1_Experiment1_code_walkthrough.md` | `_check_rgb`, both grayscale rules, cell `be37e794` |
| `A1_Experiment2_code_walkthrough.md` | `hsv_rule` incl. hue wrap, cell `c5372f53` |
| `A1_Experiment3_code_walkthrough.md` | `transform_frame`, video probe/write/verify, cell `e6418d31` |
| `A1_Task5_code_walkthrough.md` | both failures, evidence→mechanism→correction→verification |
| `A1_Task6_code_walkthrough.md` | the JPEG-vs-HSV-rule extension, cell `f390eecc` |

Each one ends with a short "what the analysis prompt is asking for" note — a structure hint
with the numbers to cite, **not** drafted prose. All nine must be deleted before packaging.

Two facts dug up while writing them, both verified and worth citing in the analysis:
- The 7.85% of the Experiment 2 target ROI the rule rejects under the *normal* capture is
  the rectangle's black outline stroke (2,401 pixels, all at value exactly 30) — not error.
- The coverage *rise* under JPEG compression is that same stroke: at q10, all 862 newly
  selected ROI pixels were outline pixels in the lossless PNG, lifted over the thresholds
  by the codec bleeding red across the black edge.

### ⏭️ WHAT IS ACTUALLY LEFT
1. **Write the analysis prose** (student's own words — deliberately not drafted): `a0e6beb4`
   (Task 2), `a0c07d3a` (Task 3), `6d7db7ff` (Task 4), `3349cb7a` (Exp 1), `baee3dde` (Exp 2),
   `64773af5` (Exp 3), `098cf7e6` (Task 5), `06afeab5` (Task 6 question + conclusion,
   80–130 words), `92d7684e` (synthesis ≤180 words), `a6e760e3` (reflection 100–150 words).
   All the numbers to cite are in the run outputs above and in `outputs/tables/`.
2. **Restart Kernel and Run All Cells**, save, export `A1.html`.
3. `validate_submission.py --assignment A1` → `package_submission.py --assignment A1`.
4. **Before packaging:** delete/gitignore all nine `A1_*_code_walkthrough.md` files at
   the repo root (study aids, not submission files):
   `rm A1_*_code_walkthrough.md` — Task1, Task2, Task3, Task4, Experiment1,
   Experiment2, Experiment3, Task5, Task6.

---

## ✅ PROGRESS (2026-09-01) — Task 3 done; Tasks 1–3 verified by a real headless run

- **Task 3 — DONE (6 pts).** Implemented `brighten_loop` + `brighten_vectorized` in
  `A1/src/student_code.py`, sharing a new `_check_uint8` guard. Both do
  **add → `np.rint` → clip `[0,255]` → `uint8`**, and both compute in **`float64`** so the
  two paths round identically on `.5` ties (this is why `outputs_identical` is a guarantee,
  not luck). Notebook cell `d58039dc` now holds the timing harness → writes
  `outputs/tables/timing.csv`.
- **Measured:** median **211 ms (loop) vs 2.89 ms (vectorized) ≈ 73×** on `wide_scene.jpg`
  (648,000 elements), offset +40, 5 repeats, outputs identical. Numbers shift per run; the
  CSV is the record.
- **Tasks 1 and 2 are now actually verified** — they had never been run under the new `A1/`.
  Headless `jupyter nbconvert --execute --allow-errors` run: setup, Task 1, Task 2, Task 3
  all execute clean; **first error is Task 4's `NotImplementedError`**, as expected.
  `image_overview.png`, `numpy_manipulations.png`, `image_metadata.csv`, `timing.csv` all
  landed in `A1/outputs/`.
- **Public tests now 5 pass / 1 skip** (was 3 pass / 3 skip).
  `test_brightness_implementations_agree` passes. The one skip is
  `test_safe_round_trip` (Task 4 stubs).
- **Task 3 verification beyond the public test:** loop == vectorized across the entire
  uint8 domain for offsets −300.5 … +300; 3-D and non-contiguous inputs; inputs not
  mutated; `uint16`/`float32` rejected; nearest-even confirmed
  (`[0,1,2,3,4] + 0.5 → [0,2,2,4,4]`).
- **New study-aid file:** `A1_Task3_code_walkthrough.md` at repo root (same style as the
  Task 2 one). **Also delete/gitignore before packaging.**

**Still TODO (student's own words, deliberately not drafted):** analysis prose for
Task 2 (cell `a0e6beb4`) and Task 3 (cell `a0c07d3a`). Both cells currently hold the
placeholder plus a `# TODO: WRITE ANALYSIS` line.

**Housekeeping found:** `__pycache__/` is in `.gitignore`, but **8 `.pyc` files are already
tracked** (the rule doesn't apply retroactively), so
`A1/src/__pycache__/student_code.cpython-311.pyc` shows up in every diff. Fix with
`git rm -r --cached` on those paths when convenient.

---

## ✅ PROGRESS (2026-08-31) — Task 1 + Task 2 applied to the canonical `A1/` starter

Picked up from the "everything reset to stubs" state and re-applied work to the new `A1/`:

- **Task 1 — DONE again.** `image_summary` re-implemented in `A1/src/student_code.py`
  (the log's reference version, lines ~194–213). The Task 1 notebook cell (`0456cb0b`) was
  already present/correct in the new notebook (uses `·` separators, `raise` commented out).
  Runs once kernel is restarted → writes `image_metadata.csv` + `image_overview.png`.
- **Task 2 — DONE (code applied).** Added `_check_rectangle` helper + implemented all six
  functions (`crop_image`, `flip_horizontal`, `extract_channel`, `bgr_to_rgb`,
  `replace_region`, `contact_sheet`) in `A1/src/student_code.py`. The Task 2 notebook cell
  (`c351ab6d`) now holds the full demo (was still the `NotImplementedError` stub) → writes
  `numpy_manipulations.png`.
- **Cell ids in the NEW notebook match the OLD ids** after all (`0456cb0b` Task 1,
  `c351ab6d` Task 2) — the earlier "new cell ids" warning did not hold; ids are unchanged.
- **Identity already filled in** the new notebook: Samantha / Salas / xso947.
- **Verified against current `A1/src/a1_tools.py`:** helpers `discover_images()`,
  `axes_grid(rows, cols)` (squeeze=False → always 2-D axes), `ensure_output_directories()`
  all exist and match how the cells call them.

**Still TODO for Task 2:** write the analysis prose in markdown cell `a0e6beb4` (student's
own words — spatial vs channel), and do a Restart-and-Run-All to confirm both figures land.
Not yet run headless this session.

**New study-aid file:** `A1_Task2_code_walkthrough.md` at repo root — a plain-markdown
explainer of the six functions + notebook cell. NOT part of the submission; **delete or
gitignore before packaging** (outputs/ must contain only required files).

**Reminder:** editing `student_code.py` does NOT affect a running kernel — Restart Kernel
(or add `%autoreload 2`) before re-running, or the old stub stays cached.

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

## STATUS AT A GLANCE (current, for the canonical `A1/` — 75 points total)

⚠️ **Point values corrected 2026-09-01** by reading the headings in `A1/A1.ipynb`. The old
table here used the deprecated `Assignment-1/` bundle's weights, which were wrong.
Notebook's own point map: core implementation 30 · required experiments 19 ·
results/analysis/failure reasoning 15 · reproducibility/submission 7 · extension 4.

| Part | Pts | Code cell | Analysis cell | State |
|---|---|---|---|---|
| Task 1 — inspect/communicate image arrays | 6 | `0456cb0b` | `6bada4c0` → prose in `fbde8d4e` | **DONE & verified** (impl + CSV + figure run clean). One grammar fix pending. |
| Task 2 — NumPy spatial/channel ops | 11 | `c351ab6d` | `a0e6beb4` | **Code DONE & verified.** Analysis prose pending. |
| Task 3 — loop vs vectorized brighten | 6 | `d58039dc` | `a0c07d3a` | **Code DONE & verified** (`timing.csv` written, ≈73×). Analysis prose pending. |
| Task 4 — dtype/range safety | 7 | `39aeb486` | `6d7db7ff` | **Code DONE & verified.** Analysis prose pending. |
| Experiment 1 — channel order / grayscale | 8 | `be37e794` | `3349cb7a` | **Code DONE & verified.** Analysis prose pending. |
| Experiment 2 — HSV rule stability | 7 | `c5372f53` | `baee3dde` | **Code DONE & verified.** Analysis prose pending. |
| Experiment 3 — video sampling | 4 | `e6418d31` | `64773af5` | **Code DONE & verified.** Analysis prose pending. |
| Task 5 — failure analysis | 7 | `4aac6891` | `098cf7e6` | **Code DONE & verified.** Analysis prose pending. |
| Task 6 — extension | 4 | `f390eecc` | `06afeab5` | **Code DONE & verified.** Question + conclusion prose pending. |
| Synthesis / reflection / submission | — | — | `92d7684e`, `a6e760e3` | Not started |

**Implemented: 15 of 15 functions (2026-09-01, later session). No stubs remain.**

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
   ~/miniforge3/bin/mamba env create -f common-setup/environment.yml
   ```
   (Verified 2026-09-01: this is the **only** `environment.yml` left in the repo.)
   The env is named `cs5243`. Note: `environment.py` requires Python **exactly
   3.11** AND importable numpy/scipy/pandas/matplotlib/PIL/cv2/skimage/sklearn/
   imageio/tqdm/torch/torchvision/yaml — so the full env (incl. pytorch) is
   required for the notebook's Section 2 to pass; a bare venv won't do.

2. **Editable-install the course package FROM THE CURRENT REPO PATH:**
   ```bash
   <path-to>/envs/cs5243/bin/pip install -e common-setup
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
~/miniforge3/envs/cs5243/bin/python -m pytest A1/tests_public -q
```
Current baseline (2026-09-01): **5 passed, 1 skipped** — the skip is `test_safe_round_trip`,
waiting on Task 4's `to_float01`/`to_uint8_safe`.

### Two different "root" helpers — do not confuse them
- `a1_tools.find_repository_root(...)` → returns **`common-setup/`** (it looks for
  `environment.yml`). This is what the public tests use.
- `cs5243.data.find_course_root(...)` → returns the **actual repo root**, which is what the
  notebook setup cell uses to build `A1 = COURSE_ROOT / "A1"`.

---

## REPO / STRUCTURE NOTES (⚠️ STALE — describes the deprecated `Assignment-1/` bundle)

> **Do not follow the paths in this section.** It predates the 2026-08-30 restructure and
> still claims the `Assignment-1/` bundle is operative. It is not — `A1/` is. Kept only as
> history. Current layout: notebook `A1/A1.ipynb` · code `A1/src/student_code.py` ·
> helpers `A1/src/a1_tools.py` (do not edit) · dataset `A1/data/{images,video}` ·
> outputs `A1/outputs/{figures,tables,videos,extension}` · tests `A1/tests_public/` ·
> scripts `common-setup/scripts/`.

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

Video `A1/data/video/moving_shapes.mp4` (was `assets/A1/video/`): 320×240, 48 frames,
12.0 fps, ~4 s. Confirmed still true by the public test on 2026-09-01.

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
- **Analysis prose**: the draft lives in a *separate* markdown cell — id **`fbde8d4e`** in
  the current `A1/A1.ipynb` (the old `ffa50a2f` was the deprecated bundle's id) —
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

## TASK 2 — CODE APPLIED & VERIFIED (11 pts)

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

**Analysis still to write** (cell `a0e6beb4`, student's own words): flip/crop/replace change
*where* pixels sit (spatial); `extract_channel`/`bgr_to_rgb` only reindex the last axis so
colors change but positions do not.

Study aid: `A1_Task2_code_walkthrough.md` (repo root, delete before packaging).

---

## TASK 3 — DONE & VERIFIED (6 pts) · cell `d58039dc` · analysis `a0c07d3a`

**Spec (from the notebook):** implement `brighten_loop` and `brighten_vectorized` using
`np.rint` nearest-even rounding **before** clipping to `[0, 255]` and converting to `uint8`.
Time each ≥3× on the same image with `time.perf_counter`; report median ms, repeat count,
and output equality in `outputs/tables/timing.csv`. "Timing supports a representation
argument; it is not a benchmarking contest."

**Key design decisions (defend these if asked):**
- Order is **add → `np.rint` → clip → cast**. The cast must come last because
  `.astype(np.uint8)` *truncates* (30.7 → 30), it does not round. (Round-vs-clip order
  happens to commute here since 0 and 255 are integers, but follow the spec's order.)
- **Both paths compute in `float64`.** The loop uses Python `float` (= C double); the
  vectorized version uses `.astype(np.float64)`. Matching precision is what makes
  "outputs identical" a guarantee rather than a coincidence — `float32` could round `.5`
  ties differently.
- `brighten_loop` flattens with `arr.reshape(-1)` so **one** loop covers 2-D grayscale and
  3-D RGB alike; writes into a separate `np.empty` output (non-mutating), reshapes at the end.
- Shared `_check_uint8` guard rejects non-`uint8` input — on the `uint16` ramp the
  `[0,255]` clip would silently crush everything to 255.
- Median (not mean) over the repeats: one sample can catch a GC pause; `max_ms` in the CSV
  shows this. CSV records min/median/max so the noise is visible.

**Verified:** loop == vectorized across the whole uint8 domain for offsets −300.5 … +300;
3-D + non-contiguous inputs; non-mutation; `uint16`/`float32` rejected; nearest-even
confirmed (`[0,1,2,3,4] + 0.5 → [0,2,2,4,4]`). Public test passes.

**Measured (2026-09-01):** median **211.4 ms loop vs 2.89 ms vectorized = 73.1×**,
`wide_scene.jpg` (300, 720, 3) = 648,000 elements, offset +40, 5 repeats, identical outputs.
Cause of the gap is **per-element Python interpreter overhead** (boxing, ufunc dispatch on
scalars, bounds checks), not better arithmetic.

**Analysis still to write** (cell `a0c07d3a`, 2–3 sentences): report the measured ratio and
explain why the comparison is controlled — same image, same offset, same repeat count, same
rounding/clipping rule; only the loop-vs-array-op strategy varies, and the equality
assertion runs before any timing is reported.

Study aid: `A1_Task3_code_walkthrough.md` (repo root, delete before packaging).

### Reference — `student_code.py` additions

```python
def _check_uint8(image: np.ndarray) -> np.ndarray:
    """Return the image as an array, rejecting anything that is not uint8."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got {arr.dtype}")
    return arr


def brighten_loop(image: np.ndarray, offset: float) -> np.ndarray:
    """Brighten uint8 values with loops, nearest-even rounding, and clipping to [0, 255]."""
    arr = _check_uint8(image)
    shift = float(offset)
    source = arr.reshape(-1)                      # one flat loop covers 2-D and 3-D alike
    result = np.empty(source.size, dtype=np.uint8)
    for i in range(source.size):
        value = float(np.rint(float(source[i]) + shift))   # round half to even, then clip
        if value < 0.0:
            value = 0.0
        elif value > 255.0:
            value = 255.0
        result[i] = value
    return result.reshape(arr.shape)


def brighten_vectorized(image: np.ndarray, offset: float) -> np.ndarray:
    """Brighten uint8 values vectorially, nearest-even rounding, and clipping to [0, 255]."""
    arr = _check_uint8(image)
    # float64 matches the loop's Python-float arithmetic, so both paths round identically
    shifted = arr.astype(np.float64) + float(offset)
    return np.clip(np.rint(shifted), 0.0, 255.0).astype(np.uint8)
```

### Reference — Task 3 notebook cell (id `d58039dc`)

```python
# STUDENT WORK — Task 3
# Controlled comparison: same image, same offset, same rounding-and-clipping rule for both
# implementations, so the only thing that varies is the loop-versus-array-op strategy.
paths = {p.name: p for p in a1_tools.discover_images()}
source = iio.imread(paths["wide_scene.jpg"])          # one fixed uint8 RGB image
OFFSET, REPEATS = 40, 5
print(f"timing on wide_scene.jpg {source.shape} {source.dtype} "
      f"({source.size:,} elements), offset=+{OFFSET}, repeats={REPEATS}")


def time_call(function, image, offset, repeats):
    """Call function(image, offset) `repeats` times; return the last result and elapsed ms."""
    samples, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(image, offset)
        samples.append((time.perf_counter() - start) * 1000.0)
    return result, samples


loop_out, loop_ms = time_call(sc.brighten_loop, source, OFFSET, REPEATS)
vector_out, vector_ms = time_call(sc.brighten_vectorized, source, OFFSET, REPEATS)

identical = bool(np.array_equal(loop_out, vector_out))
assert identical, "brighten_loop and brighten_vectorized disagree"
loop_median = float(np.median(loop_ms))
vector_median = float(np.median(vector_ms))


def timing_row(name, samples):
    """One CSV row: median/min/max milliseconds plus the controlled conditions."""
    return {
        "implementation": name, "image": "wide_scene.jpg", "shape": str(source.shape),
        "elements": int(source.size), "offset": OFFSET, "repeats": REPEATS,
        "median_ms": round(float(np.median(samples)), 4),
        "min_ms": round(min(samples), 4), "max_ms": round(max(samples), 4),
        "outputs_identical": identical,
        "speedup_vs_loop": round(loop_median / float(np.median(samples)), 2),
    }


timing = pd.DataFrame([timing_row("brighten_loop", loop_ms),
                       timing_row("brighten_vectorized", vector_ms)])
timing_path = PATHS["tables"] / "timing.csv"
timing.to_csv(timing_path, index=False)
print("wrote", timing_path)
display(timing)

# --- what the timing is an argument about: both paths share one representation rule ---
probe = np.array([[0, 10, 250]], dtype=np.uint8)
print("\nprobe uint8               :", probe.tolist())
print("brighten_loop(+20)        :", sc.brighten_loop(probe, 20).tolist())
print("brighten_vectorized(+20)  :", sc.brighten_vectorized(probe, 20).tolist())
print("naive uint8 add (+20)     :", (probe + np.uint8(20)).tolist(), "<- 250 wraps to 14")
print(f"\nmedian of {REPEATS} repeats: loop {loop_median:.2f} ms vs vectorized "
      f"{vector_median:.3f} ms = {loop_median / vector_median:.1f}x; "
      f"outputs identical: {identical}")
```

Note: the setup cell (`88492427`) already imports `time`, so the cell needs no new imports.

---

## REMAINING WORK (not started) — function + artifact checklist

All functions live in `src/student_code.py`. Cell ids are the code cell for each task.

### ~~Task 3~~ — DONE 2026-09-01, see the TASK 3 section below.

### Task 4 — `to_float01`, `to_uint8_safe` (7 pts) · cell `39aeb486` · analysis `6d7db7ff`
- `to_float01`: `uint8`/`uint16` → `float32` in `[0, 1]` (divide by the dtype max).
- `to_uint8_safe`: `float [0,1]` → clip to `[0,1]`, round, ×255, `uint8`.
- Diagnose/repair `wrong = (float_image * 255).astype(np.uint8)` (no clip → wrap).
- Compare uint8 addition vs float addition; include the uint16 ramp →
  `outputs/figures/dtype_range_experiment.png`. ≥3 quantitative comparisons →
  `outputs/tables/experiment_metrics.csv` (this CSV is cumulative — Exp 2 appends to it).
- Public test: `to_uint8_safe(to_float01(uint8_arr)) == uint8_arr`, and
  `to_float01(...).dtype == float32`.

### Experiment 1 — `grayscale_mean`, `grayscale_luminance` (8 pts) · cell `be37e794` · analysis `3349cb7a`
- `grayscale_mean`: unweighted channel mean, `float32`.
- `grayscale_luminance`: `0.299R + 0.587G + 0.114B`, `float32`.
- Compare: correct RGB display · BGR-shown-as-RGB · channel mean · luminance · OpenCV
  grayscale (with correct input convention). Fixed source image and display limits →
  `outputs/figures/color_representations.png`.

### Experiment 2 — `hsv_rule` (7 pts) · cell `c5372f53` · analysis `baee3dde`
- Boolean mask for **inclusive** OpenCV-HSV bounds, **including wrapped hue** intervals
  (when `lower[0] > upper[0]`, hue wraps around 180).
- One rule applied to `color_shapes.png` / `_dim` / `_warm` (aligned). Report
  selected-pixel fraction + one more metric per condition; **append** to
  `experiment_metrics.csv`. Figure → `outputs/figures/hsv_stability.png`.

### Experiment 3 — `transform_frame` (4 pts) · cell `e6418d31` · analysis `64773af5`
- `transform_frame`: flip RGB frame horizontally, multiply red channel by `0.65`, return
  `HxWx3 uint8`.
- `a1_tools.probe_video(...)` → record input/output/read-back structure in
  `outputs/tables/video_metadata.json`.
- Extract frames near 0.0/1.0/2.0/3.0 s → `outputs/figures/video_contact_sheet.png`.
- Apply `transform_frame` to all frames, write `outputs/videos/a1_transformed.mp4` via
  `a1_tools.write_mp4`, verify with `a1_tools.verify_video_round_trip`.

### Task 5 — failure analysis (7 pts) · cell `4aac6891` · analysis `098cf7e6`
- Two failures: one image-representation, one video/encoding. Suggested video case:
  frame index = `time * frame_count` (wrong) vs `time * FPS` (right). Show wrong result,
  explain the units error, correct + verify. Figures → `outputs/figures/failure_image.png`,
  `outputs/figures/failure_video.png`.

### Task 6 — extension (4 pts) · cell `f390eecc` · answer `06afeab5`
- One question, one controlled comparison, one principal saved output, one conclusion
  (80–130 words). Keep exactly one file in `outputs/extension/` (png/jpg/mp4/csv/json/txt).
  No new packages.

### Wrap-up
- Section 6 synthesis: cell `92d7684e` (≤180 words).
- Section 9 reflection: cell `a6e760e3` (100–150 words). Section 0 disclosure already
  lists "Claude Code".
- Submission (⚠️ paths updated 2026-09-01 — the old `Assignment-1/...` script paths were
  stale): Restart & Run All → export `A1.html` →
  `python common-setup/scripts/validate_submission.py --assignment A1` →
  `python common-setup/scripts/package_submission.py --assignment A1` →
  submit `Salas_Samantha_A1.zip`.
- **Before packaging:** delete or gitignore all nine `A1_*_code_walkthrough.md` files at
  the repo root (study aids, not submission files).

---

## VERIFYING WORK WITHOUT OPENING JUPYTER

Fast loop used on 2026-09-01, worth reusing for every remaining task.

```bash
# 1. unit-check the functions directly
~/miniforge3/envs/cs5243/bin/python -c "import sys; sys.path.insert(0,'A1/src'); import student_code as sc; ..."

# 2. public tests
~/miniforge3/envs/cs5243/bin/python -m pytest A1/tests_public -q

# 3. full headless Restart-and-Run-All (writes to a scratch copy, leaves A1.ipynb alone)
~/miniforge3/envs/cs5243/bin/jupyter nbconvert --to notebook --execute --allow-errors \
  --output /tmp/A1_executed.ipynb A1/A1.ipynb
# then read /tmp/A1_executed.ipynb with nbformat and list which cells errored
```

`--allow-errors` is essential: the run *should* stop-and-continue at the first
unimplemented task. The pass criterion is "the first error is the next task's
`NotImplementedError`," not "zero errors."

**Editing notebook cells programmatically:** use `nbformat` (read → set `cell.source`,
clear `outputs`/`execution_count` → `nbformat.validate` → write). It round-trips the file
in Jupyter's exact JSON style, so the diff stays limited to the edited cell.

**Tooling note (Claude Code on this Windows host):** the Bash tool is **Git Bash on
Windows**, not WSL — `/home/sqmi/...` paths fail there. Reach the WSL environment with
`wsl -d Ubuntu -- bash -lc '<command>'`. File read/write tools work fine against the
`\\wsl.localhost\Ubuntu\...` UNC path.

---

## TO CONTINUE ON ANOTHER DEVICE

```bash
git pull
# recreate env + editable install from common-setup/ (see ENVIRONMENT SETUP above)
# open A1/A1.ipynb, run the setup cell (88492427), then continue at Task 4 (cell 39aeb486).
# Tasks 1-3 are implemented and verified; nothing to re-paste.
```
