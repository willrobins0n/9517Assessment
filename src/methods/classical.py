from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import binary_opening, disk, remove_small_objects
from sklearn.cluster import KMeans

from src.methods.base import Segmenter

# Our first method will be lab colour space k means plant segmenters.
# For each image, we will
# 1) Convert RGB to CIE
# 2) Run k means on the a b colour channels (L is dropped by default)
# 3) Pick cluster with smallest mean as plant
# 4) Clean up


class LabKMeans(Segmenter):

    name = 'classical_lab_kmeans'

    def __init__(
        self,
        n_clusters: int = 2,
        opening_radius: int = 2,
        min_blob_size: int = 50,
        use_lightness: bool = False,
        random_state: int = 42,
    ) -> None:
    

        # We're going to have 2 clusters (plant / soil). If the plant
        # clusters leak into highlights or shadows be can bump up to 3.
        self.n_clusters = n_clusters

        #Radius of structuring disk -> we're choosing two to remove more speckle.
        # Also means we'll eat more thin leaf edges.
        self.opening_radius = opening_radius


        # We want to drop connected components that are smaller
        # than this many pixels. 
        self.min_blob_size = min_blob_size

        # Include L* in the feature vector. Usually off: L* varies
        # heavily with shadows/brightness and hurts colour clustering.

        # We include L in the feature vctor -> we can turn it off to stop it interfering
        # with shadows / brightness / colour clustering.
        self.use_lightness = use_lightness

        # Set reproducible seed so that we can get reproducable results.
        self.random_state = random_state

    def predict(self, image: np.ndarray) -> np.ndarray:

        # First we normalise the image. We want it to pass float32 RGB
        # between 0 and 1.
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        h = image.shape[0]
        w = image.shape[1]

        # Open VR expects uint8 RGB input -> output lab channels are all
        # between 0 and 255. 
        rgb_u8 = (image * 255.0)
        rgb_u8 = np.clip(rgb_u8, 0, 255)
        rgb_u8 = rgb_u8.astype(np.uint8)

        lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)

        # Now we create the feature matrix for K means. It'll be colour
        # only by default -> can add lumination if wanted.
        if self.use_lightness:
            features = lab.reshape(-1, 3)
            features = features.astype(np.float32)
        else:
            ab = lab[..., 1:]
            features = ab.reshape(-1, 2)
            features = features.astype(np.float32)
        # Now we just run our K means clustering. We choose 10 random starts
        # and keep hte best.
        km = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state,
        )

        cluster_labels = km.fit_predict(features)
        cluster_labels = cluster_labels.reshape(h, w)



        # Now we choose plant cluster -> the most negative mean a centroid
        # will be the plants.
        if self.use_lightness:
            a_col = 1
        else:
            a_col = 0

        centers = km.cluster_centers_
        plant_cluster = int(np.argmin(centers[:, a_col]))

        plant_mask = (cluster_labels == plant_cluster)

        #  Now we clean up results
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