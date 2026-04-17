# =============================================================================
# metrics.py
# -----------------------------------------------------------------------------
# Scoring primitives for binary segmentation. Pure functions — no I/O, no
# state. Used by the harness and can be called directly from notebooks for
# quick checks.
#
# Target location in the repo: src/evaluation/metrics.py
#
# The three functions build up in layers:
#   1. confusion_counts(pred, gt)     -> raw TP/FP/FN/TN pixel counts
#   2. scores_from_counts(tp, fp, fn) -> precision / recall / F1 / IoU
#   3. score_masks(pred, gt)          -> convenience wrapper: counts + scores
# =============================================================================

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# EPS — tiny positive value added to denominators to avoid division by zero.
# -----------------------------------------------------------------------------
# For example if a method predicts an all-zero mask on an image that has no
# plant pixels, TP = FP = FN = 0 and you'd otherwise hit 0/0 = NaN.
# 1e-7 is small enough not to distort any real score (precision of 0.73
# stays 0.73), and large enough to stay numerically stable.
EPS = 1e-7


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) for two binary {0, 1} masks of identical shape.

    Convention (matches EWS):
      - 1 (foreground) = plant
      - 0 (background) = soil
    """
    # -------------------------------------------------------------------------
    # Step 1: fail fast on shape mismatch.
    # -------------------------------------------------------------------------
    # This is the #1 source of silently-wrong metrics. If a method's predict()
    # returns a different HxW than the ground truth (e.g. forgot to unpad a
    # U-Net output), numpy would happily broadcast and produce meaningless
    # numbers. Raise instead.
    if pred.shape != gt.shape:
        raise ValueError(f'Shape mismatch: pred {pred.shape} vs gt {gt.shape}')

    # -------------------------------------------------------------------------
    # Step 2: cast both masks to bool.
    # -------------------------------------------------------------------------
    # Works regardless of the caller passing uint8 {0,1}, float {0.0, 1.0},
    # or already-bool. Everything non-zero becomes True.
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    # -------------------------------------------------------------------------
    # Step 3: count each confusion-matrix cell over all pixels.
    # -------------------------------------------------------------------------
    # np.logical_and gives a bool array the same shape as the inputs, True
    # wherever BOTH arguments are True. .sum() counts Trues (True == 1).
    # ~pred_b is the bitwise NOT — i.e. the background prediction.
    # int(...) converts numpy.int64 -> plain int so the CSV writer later
    # doesn't emit noisy numpy-typed values.
    tp = int(np.logical_and(pred_b, gt_b).sum())    # predicted plant & is plant
    fp = int(np.logical_and(pred_b, ~gt_b).sum())   # predicted plant & is soil
    fn = int(np.logical_and(~pred_b, gt_b).sum())   # predicted soil  & is plant
    tn = int(np.logical_and(~pred_b, ~gt_b).sum())  # predicted soil  & is soil

    return tp, fp, fn, tn


def scores_from_counts(tp: int, fp: int, fn: int, tn: int = 0) -> Dict[str, float]:
    """Convert raw pixel counts into the four required metrics.

    tn is accepted but unused — included so you can pass the full 4-tuple
    from confusion_counts() without unpacking.
    """
    # precision = of pixels we CALLED plant, how many actually ARE plant?
    precision = tp / (tp + fp + EPS)

    # recall = of pixels that ARE plant, how many did we correctly catch?
    recall = tp / (tp + fn + EPS)

    # F1 = harmonic mean of precision and recall. Goes down hard if either
    # P or R is small, so it's a good "overall" scalar for segmentation.
    f1 = 2.0 * precision * recall / (precision + recall + EPS)

    # IoU (a.k.a. Jaccard index) = intersection / union
    #                            = TP / (TP + FP + FN)
    # The standard primary metric for segmentation tasks like this one.
    iou = tp / (tp + fp + fn + EPS)

    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou}


def score_masks(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """One-call scoring used by the harness — returns metrics PLUS raw
    counts so aggregates (macro / micro) can be computed later.
    """
    tp, fp, fn, tn = confusion_counts(pred, gt)
    scores = scores_from_counts(tp, fp, fn, tn)
    # Keep raw counts around. They're useful for two things later:
    #   (a) micro averaging across images (pool counts, then score once)
    #   (b) reproducibility — you can recompute any metric from counts,
    #       but you can't go backwards from a scalar.
    scores.update({'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
    return scores
