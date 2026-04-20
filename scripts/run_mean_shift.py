#!/usr/bin/env python3
"""Run the Lab mean shift classical segmenter through the evaluation harness.

Usage:
    python scripts/run_mean_shift.py                    # evaluate on test
    python scripts/run_mean_shift.py --split val        # tune on val

Writes per-image metrics to results/mean_shift_per_image.csv
and appends one row to results/summary.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import METADATA_DIR, RESULTS_DIR
from src.evaluation.harness import evaluate
from src.methods.mean_shift import MeanShift


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Default to the test split. Switch to val while you're tuning
    # hyperparameters — only touch test for the final reported numbers.
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test'],
        default='test',
        help='Which split CSV to evaluate on.',
    )
    # Expose the most useful hyperparameters so you can sweep from the
    # shell without editing the script each time.
    parser.add_argument('--bandwidth', type=float, default=6)
    parser.add_argument('--opening-radius', type=int, default=2)
    parser.add_argument('--min-blob-size', type=int, default=50)
    parser.add_argument('--use-lightness', action='store_true')
    parser.add_argument(
        '--no-save-predictions',
        action='store_true',
        help='Skip writing prediction PNGs (faster for quick sweeps).',
    )
    args = parser.parse_args()

    seg = MeanShift(
        bandwidth=args.bandwidth,
        opening_radius=args.opening_radius,
        min_blob_size=args.min_blob_size,
        use_lightness=args.use_lightness,
    )

    split_csv = METADATA_DIR / f'{args.split}.csv'

    summary = evaluate(
        seg,
        split_csv=split_csv,
        results_dir=RESULTS_DIR,
        train_time_s=None,  # unsupervised — no training phase
        save_predictions=not args.no_save_predictions,
    )

    # One-line recap so the shell user sees the headline numbers
    # without opening the CSV.
    print(
        f"\nDone. macro IoU={summary['macro_iou']:.3f} "
        f"F1={summary['macro_f1']:.3f} "
        f"P={summary['macro_precision']:.3f} "
        f"R={summary['macro_recall']:.3f}"
    )


if __name__ == '__main__':
    main()
