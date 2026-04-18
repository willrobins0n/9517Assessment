#!/usr/bin/env python3
"""Train the U-Net deep-learning segmenter and evaluate it on a split.

Usage:
    python scripts/run_deep_unet.py                       # default: 30 epochs, evaluate on test
    python scripts/run_deep_unet.py --split val           # evaluate on val instead
    python scripts/run_deep_unet.py --num-epochs 10       # quick sanity run
    python scripts/run_deep_unet.py --no-val              # skip best-checkpoint tracking

Writes per-image metrics to results/dl_unet_r18_per_image.csv and
appends one row (with training time) to results/summary.csv.
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
from src.methods.deep_unet import UNetSegmenter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test'],
        default='test',
        help='Which split to evaluate on AFTER training.',
    )
    # Model knobs.
    parser.add_argument('--encoder', default='resnet18',
                        help='Any encoder name supported by segmentation_models_pytorch.')
    parser.add_argument('--image-size', type=int, default=352,
                        help='Must be divisible by 32 for a 5-level U-Net.')
    # Training knobs.
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--bce-weight', type=float, default=0.5)
    parser.add_argument('--dice-weight', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers. Keep 0 on small datasets / macOS.')
    parser.add_argument(
        '--no-val',
        action='store_true',
        help='Do not use val split for best-checkpoint selection. '
             'Useful if val.csv is missing or you want to train harder.',
    )
    parser.add_argument(
        '--no-save-predictions',
        action='store_true',
        help='Skip writing prediction PNGs (faster for quick sweeps).',
    )
    args = parser.parse_args()

    seg = UNetSegmenter(
        encoder_name=args.encoder,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        num_workers=args.num_workers,
    )

    # Always train on train.csv. Use val.csv for best-checkpoint
    # selection unless the user opts out.
    val_csv = None if args.no_val else METADATA_DIR / 'val.csv'
    train_time = seg.fit(METADATA_DIR / 'train.csv', val_csv=val_csv)

    split_csv = METADATA_DIR / f'{args.split}.csv'
    summary = evaluate(
        seg,
        split_csv=split_csv,
        results_dir=RESULTS_DIR,
        train_time_s=train_time,
        save_predictions=not args.no_save_predictions,
    )

    print(
        f"\nDone. macro IoU={summary['macro_iou']:.3f} "
        f"F1={summary['macro_f1']:.3f} "
        f"P={summary['macro_precision']:.3f} "
        f"R={summary['macro_recall']:.3f} "
        f"(train_time={train_time:.1f}s)"
    )


if __name__ == '__main__':
    main()
