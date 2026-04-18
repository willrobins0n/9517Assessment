"""Machine-learning segmentation with handcrafted features.

Per-pixel Random Forest classifier on a mixed colour-and-texture
feature vector. Supervised: needs the training split.

Feature vector per pixel (12 values):
    RGB (3) + HSV (3) + Lab (3) + ExG (1) + ExGR (1) + LBP (1)

Where:
    ExG  = 2G - R - B         (Excess Green vegetation index)
    ExGR = ExG - (1.4R - G)   (ExG minus Excess Red — cheap add-on)
    LBP  = uniform Local Binary Pattern on grayscale (small-scale texture)
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.morphology import binary_opening, disk, remove_small_objects
from sklearn.ensemble import RandomForestClassifier

from src.data_utils import load_image, load_mask
from src.methods.base import Segmenter


def extract_features(image: np.ndarray, lbp_radius: int = 1) -> np.ndarray:
    """Compute the per-pixel feature matrix for an RGB float32 image.

    Args:
      image: (H, W, 3) float32, values in [0, 1].
      lbp_radius: radius for the Local Binary Pattern texture feature.

    Returns:
      (H * W, F) float32 matrix of stacked per-pixel features.
    """
    h, w, _ = image.shape

    # uint8 RGB needed for OpenCV's colour-space conversions.
    rgb_u8 = (image * 255.0).clip(0, 255).astype(np.uint8)

    # HSV: OpenCV scales H to [0, 180], S/V to [0, 255]. Leaving them
    # in the raw scale is fine for a tree-based classifier — Random
    # Forest doesn't care about feature magnitudes, only orderings.
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Lab: OpenCV scales all three channels to [0, 255]. a* encodes
    # green(-)/red(+), so it's the single strongest single-channel
    # plant cue in the feature set.
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Named RGB views for readable vegetation-index formulas.
    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]

    # Excess Green (ExG, Woebbecke 1995): classic vegetation index.
    # Large positive values on green pixels, negative on red/brown soil.
    exg = 2.0 * g - r - b

    # Excess Red (ExR): 1.4R - G. ExGR = ExG - ExR amplifies plant
    # vs soil contrast compared to ExG alone. Cheap, often helpful.
    exgr = (2.0 * g - r - b) - (1.4 * r - g)

    # Local Binary Pattern on grayscale — small-scale texture.
    # 'uniform' collapses rotations of the same pattern to one code,
    # so the feature is rotation-insensitive and low-dimensional.
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    n_points = 8 * lbp_radius
    lbp = local_binary_pattern(
        gray, P=n_points, R=lbp_radius, method='uniform'
    ).astype(np.float32)

    # Stack channel-wise to (H, W, F), then flatten to (H*W, F) so one
    # row corresponds to one pixel.
    features = np.stack([
        image[..., 0], image[..., 1], image[..., 2],  # RGB (3)
        hsv[..., 0],   hsv[..., 1],   hsv[..., 2],    # HSV (3)
        lab[..., 0],   lab[..., 1],   lab[..., 2],    # Lab (3)
        exg, exgr,                                    # vegetation indices (2)
        lbp,                                          # texture (1)
    ], axis=-1).astype(np.float32)

    return features.reshape(-1, features.shape[-1])


class RandomForestSegmenter(Segmenter):
    """Random Forest pixel classifier on RGB + HSV + Lab + ExG/ExGR + LBP."""

    name = 'ml_random_forest'

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 10,
        pixels_per_image: int = 5000,
        lbp_radius: int = 1,
        opening_radius: int = 1,
        min_blob_size: int = 50,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        # Random Forest hyperparameters. Defaults favour speed over
        # squeezing the last fraction of a percent — the dataset is
        # small and inference must run on 350x350 = 122k pixels per image.
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        # Larger min_samples_leaf -> smoother boundaries, less overfit.
        self.min_samples_leaf = min_samples_leaf

        # Per-image pixel budget. Training on every pixel of every image
        # would be ~23M rows — way more than needed. Subsample.
        self.pixels_per_image = pixels_per_image

        # Feature / post-processing knobs.
        self.lbp_radius = lbp_radius
        self.opening_radius = opening_radius
        self.min_blob_size = min_blob_size

        self.random_state = random_state
        self.n_jobs = n_jobs

        # Populated by fit().
        self.clf: Optional[RandomForestClassifier] = None

    def fit(self, train_csv: Path) -> float:
        """Train the classifier on every image in `train_csv`.

        Returns wall-clock training time in seconds (feature extraction
        + RF fit), so the caller can pass it into `evaluate(...)` for
        the summary row.
        """
        train_csv = Path(train_csv)
        with train_csv.open('r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f'Empty train CSV: {train_csv}')

        rng = np.random.default_rng(self.random_state)
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        print(f'Extracting features from {len(rows)} training images...')
        t0 = time.perf_counter()

        for i, row in enumerate(rows):
            # Use the shared loaders so features line up with inference.
            image = load_image(Path(row['image_path']), color_space='rgb', normalize=True)
            mask = load_mask(Path(row['mask_path']), as_binary=True)

            feats = extract_features(image, lbp_radius=self.lbp_radius)  # (H*W, F)
            labels = mask.ravel().astype(np.int64)                       # (H*W,)

            # Stratified subsample: split the budget 50/50 between
            # plant and soil pixels so the RF doesn't learn a
            # class-imbalance shortcut. If one class is short, we fill
            # the rest from the other class.
            half = self.pixels_per_image // 2
            plant_idx = np.where(labels == 1)[0]
            soil_idx = np.where(labels == 0)[0]

            n_plant = min(half, len(plant_idx))
            n_soil = min(self.pixels_per_image - n_plant, len(soil_idx))

            chosen_plant = (
                rng.choice(plant_idx, size=n_plant, replace=False)
                if n_plant > 0 else np.array([], dtype=np.int64)
            )
            chosen_soil = (
                rng.choice(soil_idx, size=n_soil, replace=False)
                if n_soil > 0 else np.array([], dtype=np.int64)
            )
            idx = np.concatenate([chosen_plant, chosen_soil])

            x_parts.append(feats[idx])
            y_parts.append(labels[idx])

            # Progress every 20 images so the user sees life on large runs.
            if (i + 1) % 20 == 0 or (i + 1) == len(rows):
                print(f'  processed {i + 1}/{len(rows)}')

        # Concatenate across images into one big training matrix.
        x_train = np.concatenate(x_parts, axis=0)
        y_train = np.concatenate(y_parts, axis=0)
        print(
            f'Training set: X={x_train.shape}, y={y_train.shape} '
            f'(plant={int((y_train == 1).sum())}, '
            f'soil={int((y_train == 0).sum())})'
        )

        print('Training Random Forest...')
        self.clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.clf.fit(x_train, y_train)

        train_time = time.perf_counter() - t0
        print(f'Training complete in {train_time:.1f} s')
        return train_time

    def predict(self, image: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError(
                'RandomForestSegmenter.predict() called before fit(). '
                'Call .fit(train_csv) first.'
            )

        # Defensive normalisation mirroring the classical method.
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.5:
            image = image / 255.0

        h, w, _ = image.shape
        feats = extract_features(image, lbp_radius=self.lbp_radius)

        # One label per pixel; reshape back to the image grid.
        preds = self.clf.predict(feats).reshape(h, w)
        plant_mask = preds.astype(bool)

        # Same post-processing as the classical method. Pixel
        # classifiers tend to produce salt-and-pepper noise because
        # they ignore spatial context — a cheap opening cleans that up.
        if self.opening_radius > 0:
            plant_mask = binary_opening(plant_mask, disk(self.opening_radius))
        if self.min_blob_size > 0:
            plant_mask = remove_small_objects(plant_mask, min_size=self.min_blob_size)

        return plant_mask.astype(np.uint8)
