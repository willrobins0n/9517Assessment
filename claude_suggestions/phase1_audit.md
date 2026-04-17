# Phase 1 data prep audit

Review of `src/data_utils.py` and `scripts/*.py`. Ordered by severity.

---

## 1. CRITICAL — `load_mask` polarity is almost certainly inverted

**File:** `src/data_utils.py:170`

```python
if as_binary:
    array = (array < 128).astype(np.uint8)
```

This marks *dark* pixels as foreground (`1`). For the EWS dataset, the
binary masks have **plants = white (255)** and **soil = black (0)** (per
Zenkl et al. 2022, the reference paper in the spec). So this code
currently labels **soil as the positive class** — every downstream
metric (precision/recall/F1/IoU on "plant" class) would be computed
against the inverse class.

This will not crash — it will just silently flip the task. Every method
you build on top of this will appear to segment soil, not wheat.

**Suggested fix:** change `<` to `>=`:

```python
if as_binary:
    array = (array >= 128).astype(np.uint8)
```

**How to verify before committing:** run `sanity_check.py` on a handful
of training samples *after* the fix. The red overlay should sit on the
green plant pixels, not on the brown soil. If the overlay ends up on
soil, revert the change.

---

## 2. MODERATE — `summarize_samples` "binary_masks" check is a no-op

**File:** `src/data_utils.py:195-206`

```python
unique_values = np.unique(mask)
if unique_values.min() < 0 or unique_values.max() > 255:
    binary_masks = False
```

`mask` is loaded as `np.uint8` (line 168), so by construction its
values are always within `[0, 255]`. The condition can never be true,
so `binary_masks` is always reported as `True` regardless of the
actual mask content. The `dataset_summary.txt` report's
"masks binary-compatible" line is therefore meaningless.

**Suggested fix:** check whether unique values are a subset of the
allowed binary value sets. Something like:

```python
unique_set = set(unique_values.tolist())
if not (unique_set.issubset({0, 255}) or unique_set.issubset({0, 1})):
    binary_masks = False
```

Optionally, collect *which* split(s) violated, so the report can point
you at the problem files.

---

## 3. MINOR — HSV normalisation is inconsistent

**File:** `src/data_utils.py:153-163`

OpenCV HSV ranges are `H∈[0,180]`, `S∈[0,255]`, `V∈[0,255]` (not all
three in `[0, 255]`). Dividing the whole array by `255.0` therefore
squashes H to `[0, ~0.706]` while S/V go to `[0, 1]` — fine for visual
display but a bit weird as a feature representation.

This isn't a bug, but worth being aware of if you later use HSV
features in a classical ML method. Either (a) don't normalise HSV and
use raw, (b) divide H by 180 and S/V by 255 separately, or (c) use
`cv2.COLOR_BGR2HSV_FULL` which maps H to `[0, 255]`.

No action strictly required; just document whatever choice you keep.

---

## 4. MINOR — split-name inconsistency in the "combined" fallback

**File:** `src/data_utils.py:52-53`, `95-97`

If `discover_split_dirs` can't find recognised split folders, it
returns `{'all': dataset_root}`. Then inside `pair_images_and_masks`,
`split_name = canonical_split_name(root.name) or root.name.lower()` —
so `Sample.split` becomes e.g. `"ews-dataset"`, not `"all"`. The CSV
filename (from the outer loop key) says `all.csv` but each row's
`split` column says `ews-dataset`.

For the EWS dataset the three splits should be discovered normally, so
this only triggers on weird layouts. Low priority.

**Suggested fix:** pass the canonical split name into
`pair_images_and_masks` instead of re-deriving it from the directory
name — or have `pair_images_and_masks` accept an override.

---

## 5. MINOR — `summarize_samples` loads full images just to read shapes

**File:** `src/data_utils.py:195-199`

Uses `load_image(...)` / `load_mask(...)` to get `.shape`. For 190
small images this is fine, but you can get the same info from
`Image.open(path).size` (and `.mode`) without decoding pixels.

Optional perf tidy-up, not a bug.

---

## 6. MINOR — `preprocess.py` metadata scalars saved as 0-d arrays

**File:** `scripts/preprocess.py:39-48`

`np.savez_compressed(..., split=row['split'], ...)` stores each string
as a 0-d numpy array. When you `np.load(...)['split']` later, you get
a 0-d array, not a Python string — downstream code needs `.item()` to
extract it. Not broken, just a small trap.

---

## 7. NOTE — mask loading overwrites `image` local in gray branch

**File:** `src/data_utils.py:143-144`

Harmless, but it re-opens the image file. Could keep a single
`Image.open(...)` and branch on mode. Style, not correctness.

---

## Summary: what you should actually change

Two things matter for correctness:

1. **Flip the polarity in `load_mask`** (`<` → `>=`) and re-run
   `sanity_check.py` to confirm plants are the positive class.
2. **Fix the `binary_masks` check in `summarize_samples`** so the
   dataset verification report is actually informative.

Everything else is polish / awareness.
