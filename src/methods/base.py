"""Common interface for all segmentation methods.

Every method (classical, ML, DL) inherits from Segmenter so the
evaluation harness can run them through identical code paths.
"""
from __future__ import annotations

import numpy as np


class Segmenter:
    """Contract every segmentation method implements.

    Subclasses must:
      * Set `name` to a unique short string (used as a filename prefix
        in the results directory).
      * Implement `predict(image) -> mask`.

    `predict` contract:
      Input:  np.float32 RGB image of shape (H, W, 3) with values in
              [0, 1]. This is exactly what `src.data_utils.load_image`
              returns with `color_space='rgb', normalize=True`.
      Output: np.uint8 binary mask of shape (H, W) with values in
              {0, 1}. 1 = plant (foreground), 0 = soil. Must match the
              input's spatial size — no silent resizing. The harness
              raises on shape mismatch.

    Methods that need training (Random Forest, U-Net, etc.) should
    add a `fit(...)` method. `fit` is not on the base class because
    unsupervised methods (K-means, GrabCut) don't need it.
    """

    # Override in every subclass. Used as:
    #   - prefix for per-image CSV:     results/<name>_per_image.csv
    #   - folder for saved predictions: results/<name>/predictions/
    #   - value of the 'method' column in the aggregate summary.csv
    # Keep it short, snake_case, and unique across methods.
    name: str = 'base'

    def predict(self, image: np.ndarray) -> np.ndarray:
        # Fail loudly if a subclass forgets to override this, rather
        # than silently returning None / an all-zeros mask that could
        # look like a plausible (wrong) result.
        raise NotImplementedError(
            f"{type(self).__name__}.predict() is not implemented. "
            "Every Segmenter subclass must override predict()."
        )
