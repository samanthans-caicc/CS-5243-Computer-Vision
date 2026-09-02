# A1 Task 2 — Code Walkthrough

_Study note (not part of the graded submission). Explains the six NumPy functions in
`A1/src/student_code.py` and the Task 2 notebook cell._

---

## Key background: views vs. copies

The single most important NumPy idea here. When you slice an array, NumPy by default gives
you a **view** — a window onto the *same* memory, not a new array. So:

```python
a = np.array([1, 2, 3])
b = a[0:2]      # view
b[0] = 99       # ALSO changes a → a is now [99, 2, 3]
```

That's why nearly every function ends in `.copy()` — it forces a brand-new independent
array so the caller's original is never touched. The spec's **"non-mutating"** requirement
is exactly this: your functions must not alter their inputs.

---

## `_check_rectangle` — the shared validator

```python
def _check_rectangle(image, top, left, height, width):
    if min(int(top), int(left)) < 0 or min(int(height), int(width)) <= 0:
        raise ValueError(...)                      # origin must be >= 0, size must be > 0
    if top + height > image.shape[0] or left + width > image.shape[1]:
        raise ValueError(...)                      # bottom/right edge must stay inside
```

- `image.shape[0]` is height (rows), `image.shape[1]` is width (cols).
- First check: no negative origin, and width/height strictly positive (a zero-size crop is
  rejected).
- Second check: the rectangle's far edge (`top+height`, `left+width`) can't exceed the
  image. This is what makes crop/replace **"positive and in bounds."**
- It's a helper (underscore prefix = "internal") so `crop_image` and `replace_region` share
  identical validation instead of duplicating it.

---

## `crop_image`

```python
arr = np.asarray(image)
_check_rectangle(arr, top, left, height, width)
return arr[top:top + height, left:left + width].copy()
```

- `np.asarray` — cheap guarantee the input behaves like an ndarray (no copy if it already
  is one).
- `arr[top:top+height, left:left+width]` — slices rows `top…top+height` and columns
  `left…left+width`. Two slices = 2-D rectangle. If there's a channel axis, it's untouched
  (you get all channels).
- `.copy()` — returns an independent crop (non-mutating).

---

## `flip_horizontal`

```python
return np.asarray(image)[:, ::-1].copy()
```

- `[:, ::-1]` — first `:` keeps all rows; `::-1` reverses the column axis (step of −1).
  Horizontal flip = mirror left↔right, so we reverse **columns**, not rows.
- Rows stay in order (top stays top), which is why this is horizontal and not vertical.

---

## `extract_channel`

```python
if arr.ndim != 3: raise ValueError(...)
if not 0 <= channel < arr.shape[2]: raise ValueError(...)
return arr[:, :, channel].copy()
```

- `ndim != 3` guard: a grayscale image has no channel axis, so extracting a channel is
  meaningless → error.
- `arr[:, :, channel]` — all rows, all cols, **one** index on the last axis. Indexing (not
  slicing) that axis *drops* it, so a `(H, W, 3)` array becomes a 2-D `(H, W)` plane. That's
  the "return one 2-D channel" requirement.
- Channel 0 = red, 1 = green, 2 = blue (since imageio gives RGB).

---

## `bgr_to_rgb` (the "reorder" op)

```python
if arr.ndim != 3 or arr.shape[2] != 3: raise ValueError(...)
return arr[:, :, ::-1].copy()
```

- `[:, :, ::-1]` — reverse the **last** axis (channels). BGR reversed is RGB: index 0↔2
  swap, index 1 stays.
- Notice: no pixel *moves* spatially — only the per-pixel channel order changes. That's the
  "reorder/channel op" vs "spatial op" distinction from the analysis prompt.

---

## `replace_region`

```python
arr = np.asarray(image)
_check_rectangle(arr, top, left, height, width)
out = arr.copy()                                   # copy FIRST → original untouched
out[top:top + height, left:left + width] = value
return out
```

- Copies first, then writes into the copy — non-mutating.
- `out[...] = value` uses **broadcasting**: a scalar `0` fills every element; a 3-tuple like
  `(255, 0, 255)` fills each pixel's RGB with magenta. NumPy stretches the small `value` to
  fit the whole rectangle automatically.

---

## `contact_sheet` — the most involved one

Goal: tile several images into one big grid image with a border and gutters between cells.

```python
items = [np.asarray(im) for im in images]
if not items: raise ...                            # need at least one
if int(columns) <= 0: raise ...
```

**Type validation:**

```python
is_rgb = items[0].ndim == 3                         # decide grid type from first image
for im in items:
    if im.dtype != np.uint8: raise ...              # all must be uint8
    ok = (im.ndim == 3 and im.shape[2] == 3) if is_rgb else (im.ndim == 2)
    if not ok: raise ...                            # all RGB, or all grayscale — no mixing
```

This enforces the spec's "either all grayscale or all RGB uint8."

**Layout math:**

```python
border = gutter = 8
cell_h = max(im.shape[0] for im in items)           # tallest image sets cell height
cell_w = max(im.shape[1] for im in items)           # widest image sets cell width
rows = (len(items) + columns - 1) // columns        # ceil division for row count
```

- Every cell is sized to the largest image so nothing gets clipped; smaller images sit
  top-left inside their cell with `fill_value` padding around them.
- `(n + columns - 1) // columns` is the standard integer **ceiling division** trick — e.g.
  6 images in 3 columns → 2 rows; 7 images → 3 rows.

**Canvas size:**

```python
height = 2 * border + rows * cell_h + (rows - 1) * gutter
width  = 2 * border + columns * cell_w + (columns - 1) * gutter
```

- `2 * border` = 8px on each outer side. `(rows - 1) * gutter` = gutters *between* rows only
  (no gutter after the last one). Same logic for width.

```python
shape = (height, width, 3) if is_rgb else (height, width)
sheet = np.full(shape, fill_value, dtype=np.uint8)  # start filled with background color
```

**Placement:**

```python
for idx, im in enumerate(items):
    r, c = divmod(idx, columns)                     # idx → (row, col) in the grid
    y0 = border + r * (cell_h + gutter)             # top-left corner of this cell
    x0 = border + c * (cell_w + gutter)
    sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
```

- `divmod(idx, columns)` splits a flat index into `(row, col)` — e.g. idx 4, columns 3 →
  row 1, col 1.
- `y0/x0` step by one cell **plus** one gutter each time, after the initial `border` offset.
- The final assignment pastes the image at its cell's top-left; since the slice is sized to
  the *image* (`im.shape[0/1]`), not the full cell, any leftover space keeps the
  `fill_value` — that's **top-left alignment**.

---

## The notebook cell logic

- Loads the three named images, runs each op once, labeling in the comments which are
  **spatial** (crop/flip/replace) vs **channel** (extract/bgr_to_rgb).
- The `assert np.array_equal(co, before)` block **proves** non-mutation: it re-reads a
  pristine copy, runs the ops on `co`, then checks `co` is still identical. If any op
  mutated its input, this fails loudly.
- `a1_tools.axes_grid(2, 3)` makes a 2×3 figure; each `imshow` renders a result,
  `set_title` labels it, `suptitle` is the required caption, and `savefig` writes the PNG.

---

## Two concepts examiners love to probe

1. **Views vs. copies** — why `.copy()` is required for non-mutation. Be ready to say what
   happens if you *omit* it (the caller's array gets modified through the returned view).
2. **Broadcasting** — how a scalar or a 3-tuple `value` fills a whole rectangle in
   `replace_region` without a loop.
