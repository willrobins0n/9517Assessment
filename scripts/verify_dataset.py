#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import discover_split_dirs, pair_images_and_masks, summarize_samples


def main() -> None:
    parser = argparse.ArgumentParser(description='Verify EWS dataset structure and image/mask integrity.')
    parser.add_argument('dataset_root', type=Path, help='Root directory of the extracted EWS dataset.')
    parser.add_argument('--report', type=Path, default=ROOT / 'metadata' / 'dataset_summary.txt', help='Path to save the text report.')
    args = parser.parse_args()

    split_dirs = discover_split_dirs(args.dataset_root)
    lines = []
    lines.append(f'Dataset root: {args.dataset_root.resolve()}')
    lines.append(f'Discovered splits: {", ".join(split_dirs.keys())}')
    lines.append('')

    total_samples = 0
    all_good = True

    for split, split_dir in split_dirs.items():
        samples, unmatched_images, unmatched_masks = pair_images_and_masks(split_dir)
        summary = summarize_samples(samples) if samples else {'num_samples': 0, 'image_shapes': {}, 'mask_shapes': {}, 'binary_masks': False}
        total_samples += len(samples)
        if unmatched_images or unmatched_masks:
            all_good = False
        lines.append(f'[{split}]')
        lines.append(f'  directory: {split_dir.resolve()}')
        lines.append(f'  paired samples: {len(samples)}')
        lines.append(f'  unmatched images: {len(unmatched_images)}')
        lines.append(f'  unmatched masks: {len(unmatched_masks)}')
        lines.append(f'  image shapes: {summary["image_shapes"]}')
        lines.append(f'  mask shapes: {summary["mask_shapes"]}')
        lines.append(f'  masks binary-compatible: {summary["binary_masks"]}')
        if unmatched_images:
            lines.append('  examples unmatched images:')
            for p in unmatched_images[:5]:
                lines.append(f'    - {p}')
        if unmatched_masks:
            lines.append('  examples unmatched masks:')
            for p in unmatched_masks[:5]:
                lines.append(f'    - {p}')
        lines.append('')

    lines.append(f'Total paired samples: {total_samples}')
    lines.append(f'Overall status: {"PASS" if all_good else "CHECK WARNINGS ABOVE"}')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
