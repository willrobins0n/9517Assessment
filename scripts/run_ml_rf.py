from __future__ import annotations

import argparse
import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import METADATA_DIR, RESULTS_DIR
from src.evaluation.harness import evaluate
from src.methods.ml_rf import RandomForestSegmenter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test'],
        default='test',
        help='Split to evaluate on AFTER training. Train split is '
             'always used for fit().',
    )
    # Random Forest hyperparameters.
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--min-samples-leaf', type=int, default=10)
    # Training data sampling + feature knobs.
    parser.add_argument('--pixels-per-image', type=int, default=5000)
    parser.add_argument('--lbp-radius', type=int, default=1)
    # Post-processing.
    parser.add_argument('--opening-radius', type=int, default=1)
    parser.add_argument('--min-blob-size', type=int, default=50)
    parser.add_argument(
        '--no-save-predictions',
        action='store_true',
        help='Skip writing prediction PNGs (faster for quick sweeps).',
    )
    args = parser.parse_args()

    seg = RandomForestSegmenter(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        pixels_per_image=args.pixels_per_image,
        lbp_radius=args.lbp_radius,
        opening_radius=args.opening_radius,
        min_blob_size=args.min_blob_size,
    )

    # Training ALWAYS runs on train.csv — no peeking at val/test.
    train_time = seg.fit(METADATA_DIR / 'train.csv')
    
    model_dir = RESULTS_DIR / 'ml_random_forest'
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / 'rf_segmenter.pkl'
    with model_path.open('wb') as f:
        pickle.dump(seg, f)

    print(f'Saved RF model to: {model_path}')

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
