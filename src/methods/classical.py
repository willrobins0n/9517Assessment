"""Classical (unsupervised) segmentation methods.

Currently contains `LabKMeans` — a per-image K-means baseline in CIE
Lab colour space that picks the "greenest" cluster as the plant mask
and cleans up with light morphology.

This is an unsupervised baseline: it does NOT use the training set.
Good contrast to the ML and DL methods — if it does surprisingly well
it tells you the task is dominated by colour cues, and if the ML/DL
methods don't beat it by a clear margin they're not adding much.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import binary_opening, disk, remove_small_objects
from sklearn.cluster import KMeans

from src.methods.base import Segmenter


class LabKMeans(Segmenter):
    """Unsupervised Lab-colour-space K-means plant segmenter.

    Pipeline per image:
      1. Convert RGB -> CIE Lab (OpenCV).
      2. Run K-means on the a*b* colour channels (L* is dropped by
         default — lightness varies with illumination and tends to hurt
         colour clustering).
      3. Pick the cluster with the smallest mean a* as "plant"
         (most negative a* = most green in Lab).
      4. Clean up: binary opening with a small disk + remove connected
         components below `min_blob_size`.

    Hyperparameters can be tuned on the validation split.
    """

    name = 'classical_lab_kmeans'

    def __init__(
        self,
        n_clusters: int = 2,
        opening_radius: int = 2,
        min_blob_size: int = 50,
        use_lightness: bool = False,
        random_state: int = 42,
    ) -> None:
        # Number of clusters. 2 = plant/soil. Bump to 3 if the plant
        # cluster leaks into highlights or shadows — then "greenest
        # centroid" will still pick the right one.
        self.n_clusters = n_clusters

        # Radius of the structuring disk for morphological opening.
        # Larger removes more speckle at the cost of eating thin leaf
        # edges; 1-3 is the typical range for 350x350 EWS images.
        self.opening_radius = opening_radius

        # Drop connected components smaller than this many pixels.
        # Set to 0 to disable. Tune on val: too high and you lose real
        # seedlings; too low and you keep noise blobs.
        self.min_blob_size = min_blob_size

        # Include L* in the feature vector. Usually off: L* varies
        # heavily with shadows/brightness and hurts colour clustering.
        self.use_lightness = use_lightness

        # Seed so K-means results are reproducible across runs.
        self.random_state = random_state

    def predict(self, image: np.ndarray) -> np.ndarray:
        # --- Input normalisation --------------------------------------
        # Harness should pass float32 RGB in [0, 1], but be defensive.
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.5:
            # Looks like uint8 slipped through — rescale.
            image = image / 255.0

        h, w, _ = image.shape

        # --- 1. RGB -> Lab --------------------------------------------
        # OpenCV's COLOR_RGB2LAB expects uint8 RGB input. The output
        # Lab channels are all in [0, 255] (OpenCV's scaled convention).
        rgb_u8 = (image * 255.0).clip(0, 255).astype(np.uint8)
        lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)  # (H, W, 3) uint8

        # --- 2. Feature matrix for K-means ----------------------------
        # Colour-only (a*, b*) by default. Drop L* because illumination
        # changes shift L* without changing plant/soil identity.
        if self.use_lightness:
            features = lab.reshape(-1, 3).astype(np.float32)
        else:
            features = lab[..., 1:].reshape(-1, 2).astype(np.float32)

        # --- 3. Cluster ------------------------------------------------
        # n_init=10 runs K-means from 10 random starts and keeps the
        # best — standard, cheap insurance against a bad initialisation.
        km = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state,
        )
        cluster_labels = km.fit_predict(features).reshape(h, w)

        # --- 4. Pick the plant cluster --------------------------------
        # In Lab, the a* axis is green(-) <-> red(+). The most-negative
        # mean a* centroid is the greenest cluster, i.e. plants.
        # Index into cluster_centers_: a* is column 1 if we kept L*,
        # else column 0 (features are [a, b]).
        a_col = 1 if self.use_lightness else 0
        plant_cluster = int(np.argmin(km.cluster_centers_[:, a_col]))
        plant_mask = (cluster_labels == plant_cluster)

        # --- 5. Morphological cleanup ---------------------------------
        if self.opening_radius > 0:
            # Binary opening = erode then dilate. Removes small
            # isolated foreground speckles without shrinking larger
            # connected plant regions.
            plant_mask = binary_opening(plant_mask, disk(self.opening_radius))

        if self.min_blob_size > 0:
            # Drop connected components smaller than min_blob_size.
            # remove_small_objects expects a bool array and labels CCs
            # internally.
            plant_mask = remove_small_objects(plant_mask, min_size=self.min_blob_size)

        return plant_mask.astype(np.uint8)
