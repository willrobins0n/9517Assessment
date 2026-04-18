
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import discover_split_dirs, pair_images_and_masks


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'image_path', 'mask_path', 'image_name', 'mask_name'])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build split CSV indexes for the EWS dataset.')
    parser.add_argument('dataset_root', type=Path, help='Root directory of the extracted EWS dataset.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'metadata', help='Directory to save CSV indexes.')
    args = parser.parse_args()

    split_dirs = discover_split_dirs(args.dataset_root)
    combined_rows: list[dict[str, str]] = []

    for split, split_dir in split_dirs.items():
        samples, unmatched_images, unmatched_masks = pair_images_and_masks(split_dir)
        if unmatched_images or unmatched_masks:
            print(f'Warning: split={split} has unmatched files. CSV will include only matched pairs.')
        rows = [
            {
                'split': split,
                'image_path': str(sample.image_path.resolve()),
                'mask_path': str(sample.mask_path.resolve()),
                'image_name': sample.image_path.name,
                'mask_name': sample.mask_path.name,
            }
            for sample in samples
        ]
        write_csv(args.output_dir / f'{split}.csv', rows)
        combined_rows.extend(rows)
        print(f'Wrote {len(rows)} rows to {(args.output_dir / f"{split}.csv").resolve()}')

    if combined_rows:
        write_csv(args.output_dir / 'all_splits.csv', combined_rows)
        print(f'Wrote {len(combined_rows)} rows to {(args.output_dir / "all_splits.csv").resolve()}')


if __name__ == '__main__':
    main()
