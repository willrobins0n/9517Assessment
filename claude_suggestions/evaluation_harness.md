# Evaluation harness — reference draft

Three modules to create plus a config file. All go under `src/`.
Dataset path stays a placeholder for now (see `src/config.py`).

```
src/
├── config.py                   (NEW)
├── data_utils.py               (existing — already patched)
├── evaluation/
│   ├── __init__.py             (empty)
│   ├── metrics.py              (NEW)
│   └── harness.py              (NEW)
└── methods/
    ├── __init__.py             (empty)
    └── base.py                 (NEW — interface reminder)
```

Also add a `results/` folder at repo root for output CSVs (git-ignore it).

---

## `src/config.py` — one place for paths

**Purpose:** a single source of truth for paths so every script agrees
on where the dataset, metadata, and results live. The dataset root is
still a placeholder — you fill it in when you wire up the real data.

```python
from __future__ import annotations
from pathlib import Path

# TODO: point this at the extracted EWS dataset root when you wire it up.
# Keep this as the only hardcoded path in the whole project — every
# script imports it from here, so you only change one line later.
DATASET_ROOT = Path('/PATH/TO/EWS-Dataset')

# Repo root is two levels up from this file (src/config.py -> src/ -> repo/).
# Using __file__ means the path stays correct no matter where the script
# is run from (notebook, CLI, Colab, etc.).
REPO_ROOT = Path(__file__).resolve().parents[1]

# Where build_index.py writes the split CSVs.
METADATA_DIR = REPO_ROOT / 'metadata'

# Where the harness writes per-image CSVs, summary.csv, and prediction PNGs.
RESULTS_DIR = REPO_ROOT / 'results'
```

Also add `results/` to `.gitignore`.

---

## `src/evaluation/metrics.py` — scoring primitives

This module has **three functions**, all pure (no I/O, no state). They
build up from "count pixels" → "compute P/R/F1/IoU from counts" →
"score a whole predicted mask against its ground truth".

### `confusion_counts(pred, gt) -> (tp, fp, fn, tn)`

**What it does:** takes two binary masks of the same shape and counts
the four confusion-matrix cells over every pixel:

- **TP** = predicted plant AND truly plant
- **FP** = predicted plant but actually soil
- **FN** = predicted soil but actually plant
- **TN** = predicted soil AND truly soil

**How:** casts both masks to boolean, uses `np.logical_and` to get
per-pixel membership in each cell, then `.sum()` counts the `True`s.
`int(...)` converts the numpy scalar to a plain Python int so it
serialises cleanly into CSV later. The shape check at the top fails
fast if the predicted mask has a different size to the ground truth
(catches resizing bugs).

### `scores_from_counts(tp, fp, fn, tn) -> dict`

**What it does:** converts raw counts into the four metrics the spec
asks for (precision, recall, F1, IoU). The formulas are standard:

- precision = TP / (TP + FP) — of pixels we called plant, how many
  actually are?
- recall = TP / (TP + FN) — of pixels that are plant, how many did
  we catch?
- F1 = 2·P·R / (P + R) — harmonic mean of P and R.
- IoU = TP / (TP + FP + FN) — overlap over union; the main metric
  for segmentation.

**Why the `+ EPS`:** if a mask has no foreground (or no prediction)
you'd divide by zero. Adding `1e-7` is a standard dodge; it also
nudges the output to roughly 0 in that case instead of `NaN`.

### `score_masks(pred, gt) -> dict`

**What it does:** convenience wrapper that chains the two above —
count pixels, convert to scores, and bundle both the scores *and* the
raw counts in one dict. The harness uses this so each image yields a
single per-image row.

```python
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# Tiny positive value added to denominators so we never divide by zero
# when a mask or prediction happens to be empty. 1e-7 is small enough
# not to shift any meaningful score.
EPS = 1e-7


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) for two binary {0,1} masks of identical shape."""
    # Guard against the #1 source of bad metrics: a prediction whose
    # shape doesn't match the ground truth. Fail loud instead of
    # silently broadcasting or resizing.
    if pred.shape != gt.shape:
        raise ValueError(f'Shape mismatch: pred {pred.shape} vs gt {gt.shape}')

    # Cast to bool so logical ops work element-wise regardless of
    # whether the caller passed uint8 {0,1}, float {0.0, 1.0}, or bool.
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    # Each cell of the confusion matrix is an element-wise AND of the
    # appropriate mask / negated mask, then a total count of True pixels.
    # int(...) converts numpy.int64 to a plain Python int so the CSV
    # writer downstream doesn't add "np.int64(...)" noise.
    tp = int(np.logical_and(pred_b, gt_b).sum())    # plant called plant
    fp = int(np.logical_and(pred_b, ~gt_b).sum())   # soil called plant
    fn = int(np.logical_and(~pred_b, gt_b).sum())   # plant called soil
    tn = int(np.logical_and(~pred_b, ~gt_b).sum())  # soil called soil
    return tp, fp, fn, tn


def scores_from_counts(tp: int, fp: int, fn: int, tn: int = 0) -> Dict[str, float]:
    # Standard per-class scoring formulas. EPS prevents NaN when a
    # denominator would otherwise be zero.
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    # F1 is the harmonic mean of precision and recall.
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    # IoU = intersection / union. Equivalently TP / (TP + FP + FN).
    iou = tp / (tp + fp + fn + EPS)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou}


def score_masks(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    # One call the harness uses per image: returns metrics + raw counts
    # bundled so we can both report P/R/F1/IoU AND aggregate counts
    # later for micro-averaged scores.
    tp, fp, fn, tn = confusion_counts(pred, gt)
    scores = scores_from_counts(tp, fp, fn, tn)
    scores.update({'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
    return scores
```

---

## `src/methods/base.py` — the interface every method implements

**Purpose:** a tiny class that fixes the contract between "a
segmentation method" and "the harness". It doesn't do any work
itself — it's basically documentation in code form. By having each of
the three methods inherit this, the harness can call `.predict(image)`
without knowing or caring whether it's K-means, Random Forest, or
U-Net underneath.

**The contract:**

- `name` — a unique string used for filenames (e.g.
  `classical_lab_kmeans`, `ml_rf`, `dl_unet_r18`). The harness uses
  this to name the per-image CSV and the predictions folder.
- `predict(image)` — takes an RGB float32 array shape `(H, W, 3)` with
  values in `[0, 1]` and returns a binary uint8 array shape `(H, W)`
  with values in `{0, 1}`. **Same H, W going in and out** — no
  silent resizing; the harness will complain if you break this.

```python
from __future__ import annotations

import numpy as np


class Segmenter:
    """A segmentation method the harness can evaluate.

    Subclasses must set `name` and implement `predict`. `predict` must
    return a binary {0, 1} uint8 mask of the SAME (H, W) as the input
    image. Input is an RGB float32 array in [0, 1] of shape (H, W, 3).
    """

    # Unique string per method. Used as a filename prefix for
    # per-image CSVs and as the folder name for saved predictions.
    # Override in every subclass (e.g. 'classical_lab_kmeans').
    name: str = 'base'

    def predict(self, image: np.ndarray) -> np.ndarray:
        # Every concrete method must override this. Fail loudly if
        # anyone forgets — easier to debug than a silent all-zeros
        # baseline that looks plausible.
        raise NotImplementedError
```

Methods that need training (Random Forest, U-Net) will add a `fit`
method too — it isn't part of the base contract because unsupervised
methods (K-means) don't need it.

---

## `src/evaluation/harness.py` — the `evaluate` function

This is the whole point of the module. One function, called once per
method per split, which produces the numbers that end up in the
report.

### What `evaluate(segmenter, split_csv, results_dir, ...)` does, step by step

**1. Read the split CSV.** The split CSV (produced by `build_index.py`
in Phase 1) has one row per sample with `image_path` and `mask_path`
columns. The function reads the whole list into memory — it's 190
images max, so this is cheap.

**2. Make a predictions folder** if `save_predictions=True`. This
writes one PNG per input image, which you'll need for the qualitative
figures in the report (successful + failure examples).

**3. Loop over every sample, and for each one:**

- Load the RGB image normalised to `[0, 1]` using the shared
  `load_image` helper.
- Load the ground-truth binary mask using the shared `load_mask`
  helper (so every method sees the *same* GT).
- Start a `time.perf_counter()` stopwatch, call `segmenter.predict`,
  stop the clock. This is the inference time for this image.
- Squeeze any singleton dimensions out of the prediction and cast to
  uint8 (some methods — e.g. U-Net — return `(1, H, W)` or `(H, W, 1)`;
  this normalises it).
- Score it against the ground truth with `score_masks` from
  `metrics.py`.
- Append a per-image row (name, P, R, F1, IoU, time, raw counts).
- Accumulate running totals of TP/FP/FN/TN so we can also compute
  "micro" scores at the end.
- Optionally save the predicted mask as a PNG (multiplied by 255 so
  it's visible).
- Print a one-line progress update.

**4. Write the per-image CSV** — one row per test image, all metrics,
so the report can show distributions / find worst examples / etc.

**5. Compute two kinds of aggregate** at the end:

- **Macro** = mean of per-image metrics. This treats every image
  equally regardless of how much foreground it has.
- **Micro** = pool all pixel counts across images, then compute
  metrics once on the totals. This is dominated by images with more
  foreground pixels.

Both are standard. The report's main table should use macro (matches
"compute metric per image, then average"). Micro is useful as a
sanity check in discussion — if they disagree by a lot, it tells you
the method performs very differently on small-plant vs big-plant
images.

**6. Append one row to `results/summary.csv`.** This is the single
unified table: each method writes one row, and once all three have
run you have exactly the comparison table you need for the report.
Header is written only the first time the file is created.

**7. Return the summary dict** so you can `print` / inspect it
inline.

### Why `train_time_s` is a kwarg, not measured inside the harness

Training happens outside this function (each method has its own
`fit` or training script). Measure it there with
`time.perf_counter()`, then pass the number into `evaluate(...)` so
it appears in the same summary row. The harness only times *inference*
per image.

### What to watch for

- If a method's `predict` returns a different spatial shape to the
  input, `score_masks` will raise. This is deliberate — resizing
  silently would corrupt metrics.
- `save_predictions=True` writes one PNG per image. For 50 test images
  that's fine. If you later blow out the dataset size, consider
  turning it off and only saving a hand-picked subset.

```python
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Make `src.*` imports work when this file is run directly or from
# a notebook. Walking up two parents gets us to the repo root
# (src/evaluation/harness.py -> src/evaluation/ -> src/ -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_utils import load_image, load_mask
from src.evaluation.metrics import score_masks, scores_from_counts
from src.methods.base import Segmenter


def evaluate(
    segmenter: Segmenter,
    split_csv: Path,
    results_dir: Path,
    train_time_s: Optional[float] = None,
    save_predictions: bool = False,
    verbose: bool = True,
) -> dict:
    """Run `segmenter.predict` over every sample in `split_csv` and
    write per-image + summary metrics to `results_dir`.

    Returns the summary dict (also appended to results/summary.csv).
    """
    # --- Setup -----------------------------------------------------------
    split_csv = Path(split_csv)
    results_dir = Path(results_dir)
    # exist_ok=True -> fine to call on every run without clearing first.
    results_dir.mkdir(parents=True, exist_ok=True)

    # Read the whole split CSV up front. Small (<=190 rows for EWS), so
    # no streaming needed.
    with split_csv.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty split CSV: {split_csv}')

    # Per-method subfolder for predicted mask PNGs. Only created if the
    # caller asked for them (they're useful for qualitative figures).
    pred_dir = results_dir / segmenter.name / 'predictions'
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    # --- Accumulators ----------------------------------------------------
    # One dict per image -> becomes the per-image CSV.
    per_image = []
    # Running pixel-count totals -> used at the end for micro averages.
    total_tp = total_fp = total_fn = total_tn = 0
    # Sum of per-image inference times -> reported in summary row.
    total_infer = 0.0

    # --- Main loop -------------------------------------------------------
    for row in rows:
        # Load inputs using the same helpers as every other script so
        # colour space / normalisation / polarity is identical everywhere.
        # image: float32 (H, W, 3) in [0, 1]
        # gt:    uint8   (H, W)    in {0, 1}
        image = load_image(Path(row['image_path']), color_space='rgb', normalize=True)
        gt = load_mask(Path(row['mask_path']), as_binary=True)

        # Time just the prediction call — NOT I/O. perf_counter is the
        # highest-resolution clock Python exposes; correct for timing
        # short durations like a single image.
        t0 = time.perf_counter()
        pred = segmenter.predict(image)
        infer = time.perf_counter() - t0
        total_infer += infer

        # Normalise prediction shape: some methods return (H, W, 1) or
        # (1, H, W); squeeze collapses those singletons. Then cast to
        # uint8 so confusion_counts gets a clean {0, 1} mask.
        pred = np.asarray(pred).squeeze().astype(np.uint8)
        # One-call scoring + raw counts for later aggregation.
        scores = score_masks(pred, gt)

        # Build the per-image row. Keeping tp/fp/fn/tn lets you
        # recompute any derived metric later without re-running.
        per_image.append({
            'image_name': row['image_name'],
            'precision': scores['precision'],
            'recall': scores['recall'],
            'f1': scores['f1'],
            'iou': scores['iou'],
            'infer_time_s': infer,
            'tp': scores['tp'], 'fp': scores['fp'],
            'fn': scores['fn'], 'tn': scores['tn'],
        })
        # Accumulate pixel totals for the micro-average computation below.
        total_tp += scores['tp']; total_fp += scores['fp']
        total_fn += scores['fn']; total_tn += scores['tn']

        if save_predictions:
            # Multiply by 255 so 0/1 masks are visible as black/white PNGs.
            Image.fromarray((pred * 255).astype(np.uint8)).save(
                pred_dir / f"{Path(row['image_name']).stem}.png"
            )
        if verbose:
            # One-line progress so you can tell the loop is alive and
            # spot any obviously-broken prediction early.
            print(f"{row['image_name']}: IoU={scores['iou']:.3f} "
                  f"F1={scores['f1']:.3f} ({infer * 1000:.1f} ms)")

    # --- Per-image CSV ---------------------------------------------------
    # Keyed on the segmenter name so multiple methods don't clobber one
    # another's per-image file.
    per_image_csv = results_dir / f'{segmenter.name}_per_image.csv'
    with per_image_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    # --- Aggregates ------------------------------------------------------
    # Macro = mean of per-image metrics. Treats every image equally
    # regardless of how many plant pixels it has. Matches the spec's
    # "compute metric per image, then average" phrasing — use this
    # for the main results table in the report.
    macro = {k: float(np.mean([r[k] for r in per_image]))
             for k in ['precision', 'recall', 'f1', 'iou']}

    # Micro = pool all pixels across all images, then compute once.
    # Dominated by images with lots of plant. If macro and micro
    # disagree a lot, it's a signal your method behaves very
    # differently on sparse-plant vs dense-plant images — worth a
    # sentence in the discussion section.
    micro = scores_from_counts(total_tp, total_fp, total_fn, total_tn)

    # --- Summary row -----------------------------------------------------
    summary = {
        'method': segmenter.name,
        'split_csv': str(split_csv),
        'n_images': len(per_image),
        'macro_precision': macro['precision'],
        'macro_recall': macro['recall'],
        'macro_f1': macro['f1'],
        'macro_iou': macro['iou'],
        'micro_precision': micro['precision'],
        'micro_recall': micro['recall'],
        'micro_f1': micro['f1'],
        'micro_iou': micro['iou'],
        'total_infer_time_s': total_infer,
        'mean_infer_time_s': total_infer / len(per_image),
        # Training time is measured by the caller (outside this harness).
        # Empty string keeps the CSV cell visibly empty for methods that
        # don't train (e.g. K-means).
        'train_time_s': train_time_s if train_time_s is not None else '',
    }

    # Append-mode on summary.csv so each method adds one row. Write the
    # header only the first time the file is created.
    summary_csv = results_dir / 'summary.csv'
    write_header = not summary_csv.exists()
    with summary_csv.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    return summary
```

---

## How a method plugs into this

The interface from the method's side is dead simple — just implement
`predict`. Example shape:

```python
from src.methods.base import Segmenter

class LabKMeans(Segmenter):
    name = 'classical_lab_kmeans'

    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters

    def fit(self, train_csv):  # optional — only if the method trains
        ...

    def predict(self, image):
        # image: float32 RGB in [0, 1], shape (H, W, 3)
        # return: uint8 {0, 1}, shape (H, W)
        ...
```

And the end-of-script call that does the actual evaluation:

```python
from src.evaluation.harness import evaluate
from src.config import METADATA_DIR, RESULTS_DIR

seg = LabKMeans(n_clusters=2)
# seg.fit(METADATA_DIR / 'train.csv')  # if applicable
summary = evaluate(seg, METADATA_DIR / 'test.csv', RESULTS_DIR,
                   train_time_s=None, save_predictions=True)
print(summary)
```

After running all three methods, `results/summary.csv` will have
three rows — paste straight into the report.
