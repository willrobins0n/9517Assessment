from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import binary_opening, disk, remove_small_objects
import sklearn.cluster as skl_clu
from joblib import parallel_backend

from src.methods.base import Segmenter

class MeanShift(Segmenter):

    name = 'mean_shift_segmenter'

    def __init__(
        self,
        bandwidth: float = 6,
        opening_radius: int = 1,
        min_blob_size: int = 20,
        use_lightness: bool = False
    ) -> None:

        self.bandwidth = bandwidth
        self.opening_radius = opening_radius
        self.min_blob_size = min_blob_size
        self.use_lightness = use_lightness

    def predict(self, image: np.ndarray) -> np.ndarray:

        if image.dtype != np.float32:
            image = image.astype(np.float32)

        h, w, _ = image.shape

        rgb_u8 = (image * 255.0)
        rgb_u8 = np.clip(rgb_u8, 0, 255)
        rgb_u8 = rgb_u8.astype(np.uint8)

        lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)

        if self.use_lightness:
            features = lab.reshape(-1, 3)
            features = features.astype(np.float32)
        else:
            lab = lab[..., 1:]
            features = lab.reshape(-1, 2)
            features = features.astype(np.float32)

        ms = skl_clu.MeanShift(
            bandwidth=self.bandwidth,
            bin_seeding=True,
        )

        with parallel_backend('threading', n_jobs=-3):
            cluster_labels = ms.fit_predict(features)

        cluster_labels = cluster_labels.reshape((h, w))

        # Remove excess labels
        if self.use_lightness:
            a_col = 1
        else:
            a_col = 0

        centers = ms.cluster_centers_
        plant_cluster = int(np.argmin(centers[:, a_col]))

        plant_mask = (cluster_labels == plant_cluster)

        if self.opening_radius > 0:

            # We remove small isolated speckles without shrinking the larger
            # connected plant regions.
            selem = disk(self.opening_radius)
            plant_mask = binary_opening(plant_mask, selem)

        if self.min_blob_size > 0:

            # now we drop the connected components that are smaller than
            # minimum blob size. 
            plant_mask = remove_small_objects(
                plant_mask,
                min_size=self.min_blob_size
            )

        return plant_mask.astype(np.uint8)
