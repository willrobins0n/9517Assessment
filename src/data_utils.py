from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MASK_HINTS = {'mask', 'masks', 'label', 'labels', 'annotation', 'annotations', 'gt', 'groundtruth'}
SPLIT_ALIASES = {
    'train': ['train', 'training'],
    'val': ['val', 'valid', 'validation', 'dev'],
    'test': ['test', 'testing'],
}


@dataclass(frozen=True)
class Sample:
    split: str
    image_path: Path
    mask_path: Path


def _contains_hint(path: Path, hints: Sequence[str]) -> bool:
    lower_parts = [p.lower() for p in path.parts]
    stem = path.stem.lower()
    return any(h in lower_parts for h in hints) or any(h in stem for h in hints)


def canonical_split_name(name: str) -> Optional[str]:
    lower = name.lower()
    for canonical, aliases in SPLIT_ALIASES.items():
        if lower in aliases:
            return canonical
    return None


def discover_split_dirs(dataset_root: Path) -> Dict[str, Path]:
    dataset_root = Path(dataset_root)
    split_dirs: Dict[str, Path] = {}
    for path in dataset_root.rglob('*'):
        if not path.is_dir():
            continue
        canonical = canonical_split_name(path.name)
        if canonical and canonical not in split_dirs:
            split_dirs[canonical] = path
    if split_dirs:
        return split_dirs

    # Fallback: no explicit split directories found. Treat root as a combined source.
    return {'all': dataset_root}


def list_candidate_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def is_mask_file(path: Path) -> bool:
    return _contains_hint(path, tuple(MASK_HINTS)) or path.stem.lower().endswith('_mask')


def is_image_file(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_EXTS:
        return False
    if is_mask_file(path):
        return False
    return True


def normalize_stem(stem: str) -> str:
    value = stem.lower()
    for suffix in ['_mask', '-mask', ' mask', '_label', '-label', ' label', '_gt', '-gt', ' gt', '_annotation', '-annotation']:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    value = value.replace(' ', '').replace('-', '').replace('_', '')
    return value


def split_image_and_mask_files(files: Iterable[Path]) -> Tuple[List[Path], List[Path]]:
    images: List[Path] = []
    masks: List[Path] = []
    for f in files:
        if is_mask_file(f):
            masks.append(f)
        else:
            images.append(f)
    return sorted(images), sorted(masks)


def pair_images_and_masks(root: Path) -> Tuple[List[Sample], List[Path], List[Path]]:
    root = Path(root)
    split_name = canonical_split_name(root.name) or root.name.lower()
    files = list_candidate_files(root)
    images, masks = split_image_and_mask_files(files)

    mask_by_stem: Dict[str, List[Path]] = {}
    for mask in masks:
        key = normalize_stem(mask.stem)
        mask_by_stem.setdefault(key, []).append(mask)

    samples: List[Sample] = []
    unmatched_images: List[Path] = []
    matched_mask_paths: set[Path] = set()

    for image in images:
        key = normalize_stem(image.stem)
        candidates = mask_by_stem.get(key, [])
        if not candidates:
            # fallback: look for mask in same folder with *_mask suffix
            local_candidate = image.with_name(f"{image.stem}_mask.png")
            if local_candidate.exists():
                candidates = [local_candidate]
        if not candidates:
            unmatched_images.append(image)
            continue

        def candidate_score(mask: Path) -> Tuple[int, int, int]:
            score_same_parent = int(mask.parent == image.parent)
            score_mask_dir = int(_contains_hint(mask.parent, tuple(MASK_HINTS)))
            common_prefix = len(set(image.parts) & set(mask.parts))
            return (score_same_parent, score_mask_dir, common_prefix)

        best = sorted(candidates, key=candidate_score, reverse=True)[0]
        samples.append(Sample(split=split_name, image_path=image, mask_path=best))
        matched_mask_paths.add(best)

    unmatched_masks = [m for m in masks if m not in matched_mask_paths]
    return sorted(samples, key=lambda s: s.image_path.name), unmatched_images, unmatched_masks


def load_image(image_path: Path, color_space: str = 'rgb', normalize: bool = True) -> np.ndarray:
    image = Image.open(image_path).convert('RGB')
    array = np.asarray(image, dtype=np.float32)
    color_space = color_space.lower()
    if color_space == 'rgb':
        pass
    elif color_space == 'gray':
        image = Image.open(image_path).convert('L')
        array = np.asarray(image, dtype=np.float32)[..., None]
    elif color_space in {'hsv', 'lab'}:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                f"OpenCV is required for color_space='{color_space}'. Install opencv-python."
            ) from exc
        bgr = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2BGR)
        if color_space == 'hsv':
            converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        else:
            converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        array = converted.astype(np.float32)
    else:
        raise ValueError(f'Unsupported color_space: {color_space}')

    if normalize:
        array /= 255.0
    return array


def load_mask(mask_path: Path, as_binary: bool = True) -> np.ndarray:
    mask = Image.open(mask_path).convert('L')
    array = np.asarray(mask, dtype=np.uint8)
    if as_binary:
        array = (array < 128).astype(np.uint8)
    return array


def basic_mask_stats(mask: np.ndarray) -> Dict[str, object]:
    unique_values = np.unique(mask)
    return {
        'shape': tuple(mask.shape),
        'unique_values': unique_values.tolist(),
        'is_binary': bool(set(unique_values.tolist()).issubset({0, 1, 255})),
        'foreground_ratio': float((mask > 0).mean()),
    }


def summarize_samples(samples: Sequence[Sample]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        'num_samples': len(samples),
        'image_shapes': {},
        'mask_shapes': {},
        'binary_masks': True,
    }
    image_shapes: Dict[Tuple[int, ...], int] = {}
    mask_shapes: Dict[Tuple[int, ...], int] = {}
    binary_masks = True

    for sample in samples:
        img = load_image(sample.image_path, color_space='rgb', normalize=False)
        mask = load_mask(sample.mask_path, as_binary=False)
        image_shapes[tuple(img.shape)] = image_shapes.get(tuple(img.shape), 0) + 1
        mask_shapes[tuple(mask.shape)] = mask_shapes.get(tuple(mask.shape), 0) + 1
        unique_values = np.unique(mask)
        if unique_values.min() < 0 or unique_values.max() > 255:
            binary_masks = False

    summary['image_shapes'] = {str(k): v for k, v in image_shapes.items()}
    summary['mask_shapes'] = {str(k): v for k, v in mask_shapes.items()}
    summary['binary_masks'] = binary_masks
    return summary
