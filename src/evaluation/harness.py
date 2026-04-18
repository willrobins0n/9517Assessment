

# All 3 of my methods will hit the same data loading code, the same scoring code
# and the same aggregation. As such, the numbers are directly comparable.

# Well have one entry point that compares all three methods.
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
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
    split_csv = Path(split_csv)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # We're going to load the entire CSV up front, as the dataset is
    # quite small so should be ok.
    with split_csv.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty split CSV: {split_csv}')


    # We'll have a predictions folder per method. However, we only want it to
    # be created if we're wriitng PNGs.
    pred_dir = results_dir / segmenter.name / 'predictions'
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    # Per-image rows become the method-specific CSV at the end.
    per_image: list[dict] = []



    # We result the pool across all images, as well as the sum of the per image
    # inference times.
    total_tp = total_fp = total_fn = total_tn = 0
    total_infer = 0.0

    for row in rows:
        # Load inputs via the shared helpers so every method sees the
        # same RGB range, the same polarity convention, the same
        # everything. This is what keeps results comparable.
        image = load_image(Path(row['image_path']), color_space='rgb', normalize=True)
        gt = load_mask(Path(row['mask_path']), as_binary=True)

        # Time the predict() call only — not I/O. perf_counter is the
        # highest-resolution clock Python exposes; accurate for short
        # per-image durations.

        # We want to only time the predict() call, not the I/O.
        t0 = time.perf_counter()
        pred = segmenter.predict(image)
        infer = time.perf_counter() - t0
        total_infer += infer

        # Normalise prediction shape and dtype. Some methods return
        # (1, H, W) or (H, W, 1); squeeze() collapses those singletons
        # so score_masks() sees a clean (H, W) uint8 mask.

        # Now we need to normalise prediction shape and dtype. Some methods
        # return (1, H, W), some return (H, W, 1). We can collapse them using
        # the squeeze() function. 
        pred = np.asarray(pred).squeeze().astype(np.uint8)
        scores = score_masks(pred, gt)

        per_image.append({
            'image_name': row['image_name'],
            'precision': scores['precision'],
            'recall': scores['recall'],
            'f1': scores['f1'],
            'iou': scores['iou'],
            'infer_time_s': infer,
            # Keep raw counts in the CSV so any derived metric can be
            # recomputed later without re-running the whole evaluation.
            'tp': scores['tp'], 'fp': scores['fp'],
            'fn': scores['fn'], 'tn': scores['tn'],
        })

        total_tp += scores['tp']
        total_fp += scores['fp']
        total_fn += scores['fn']
        total_tn += scores['tn']

        if save_predictions:
            # Multiply by 255 so a 0/1 mask is visible as a black/white PNG.
            Image.fromarray((pred * 255).astype(np.uint8)).save(
                pred_dir / f"{Path(row['image_name']).stem}.png"
            )

        if verbose:
            print(
                f"{row['image_name']}: "
                f"IoU={scores['iou']:.3f} F1={scores['f1']:.3f} "
                f"({infer * 1000:.1f} ms)"
            )

    # We can have the CSV per image (one row for every test image)
    per_image_csv = results_dir / f'{segmenter.name}_per_image.csv'
    with per_image_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    # The macro is the mean of the per-image metrics. We treat every image
    # equally, reagardless of the size of the foreground. This is what we'll
    # use for the main results table.
    macro = {
        k: float(np.mean([r[k] for r in per_image]))
        for k in ('precision', 'recall', 'f1', 'iou')
    }

    # The micro is the pool counts across all of the images (where metrics are
    # computed once). It'll be dominated by the images that have more plant pixels.
    micro = scores_from_counts(total_tp, total_fp, total_fn, total_tn)

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
        # Training time is measured by the caller (outside the harness).
        # Empty cell for methods that don't train (e.g. K-means).
        'train_time_s': '' if train_time_s is None else train_time_s,
    }

    # Append one row to the shared summary.csv. Write the header only
    # if the file doesn't exist yet, so each subsequent method's
    # evaluate() call just adds another comparison row.
    summary_csv = results_dir / 'summary.csv'
    write_header = not summary_csv.exists()
    with summary_csv.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    return summary
