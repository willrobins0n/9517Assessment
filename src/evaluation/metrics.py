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


# This will be used for scoring primitives by binary segmentation.
# We need three functions:
# 1 -> compute the raw pixel counts for TP FP FN TN.
# 2 -> compute the scores from count -> compute precision, recall, F1, IoU
# 3 -> score masks -> convenience wrapper: counts + scores.
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# We're going to define some small number, epsilon. I'm finding that
# if a method predicts an all zero mask, then we get 0 / 0 = NaN.
# 1e-7 is small enough to not distort any real score.



EPS = 1e-7


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:

    # Fail fast if the prediction's shape doesn't match the ground truth.
    # Without this, numpy would silently broadcast and produce meaningless
    # metrics.
    if pred.shape != gt.shape:
        raise ValueError(f'Shape mismatch: pred {pred.shape} vs gt {gt.shape}')

    # We need to make both masks a bool.
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)


    # Now, we get the confusion matrix cell over all of the pixels.
    tp = int(np.logical_and(pred_bool, gt_bool).sum())  
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())  
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())     
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())    

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

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


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
    scores.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return scores