# =============================================================================
# base.py
# -----------------------------------------------------------------------------
# The contract every segmentation method must satisfy so the evaluation
# harness can run it. This file is basically documentation-in-code.
#
# Target location in the repo: src/methods/base.py
#
# Why a base class and not just a protocol / duck typing:
#   - Makes the contract obvious at the top of every method file:
#         class LabKMeans(Segmenter): ...
#     immediately tells the reader this thing plugs into the harness.
#   - Gives you one place to raise NotImplementedError if someone forgets
#     to implement predict(), which is much easier to debug than a silent
#     all-zeros baseline.
#   - If you later want to add shared helpers (e.g. common preprocessing),
#     this is the natural place.
# =============================================================================

from __future__ import annotations

import numpy as np


class Segmenter:
    """A segmentation method the harness can evaluate.

    Subclasses must:
      - Set `name` to a unique short string (used for filenames).
      - Implement `predict(image) -> mask`.

    `predict` contract:
      Input:
        image : np.ndarray, dtype float32, shape (H, W, 3), values in [0, 1].
                RGB colour order. This is what src.data_utils.load_image
                returns with color_space='rgb', normalize=True.

      Output:
        mask  : np.ndarray, dtype uint8, shape (H, W), values in {0, 1}.
                1 = plant (foreground), 0 = soil (background).
                SAME (H, W) as the input image — no silent resizing.
                The harness will raise if shapes don't match.

    Methods that need training (Random Forest, U-Net) will add a `fit`
    method too. `fit` is NOT part of the base contract because unsupervised
    methods (K-means, GrabCut) don't need it.
    """

    # -------------------------------------------------------------------------
    # name — override this in every concrete subclass.
    # -------------------------------------------------------------------------
    # Used by the harness as:
    #   - filename prefix for the per-image CSV:
    #         results/<name>_per_image.csv
    #   - folder name for saved prediction PNGs:
    #         results/<name>/predictions/
    #   - the 'method' column in results/summary.csv (main comparison table)
    #
    # Keep it short and snake_case and UNIQUE across methods, e.g.:
    #     name = 'classical_lab_kmeans'
    #     name = 'ml_rf'
    #     name = 'dl_unet_r18'
    name: str = 'base'

    def predict(self, image: np.ndarray) -> np.ndarray:
        # Every concrete method must override this. If someone forgets, we
        # fail loudly on the first call rather than silently returning None
        # or an all-zeros mask that looks like a plausible-but-wrong result.
        raise NotImplementedError(
            f"{type(self).__name__}.predict() is not implemented. "
            "Every subclass of Segmenter must override predict()."
        )
