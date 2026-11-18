"""
This module isn't actually used in UMF, just adding it here to make loading CS-WP-MM training
tuples easier.
"""
from typing import Dict, List

import numpy as np


class TrainingTuple:
    # Tuple describing an element for training/validation, with optional pose for metric localisation
    def __init__(
        self,
        id: int,
        timestamp: float,
        rel_scan_filepath: str,
        positives: np.ndarray,
        non_negatives: np.ndarray,
        position: np.ndarray,
        pose: np.ndarray = None,
        positives_poses: Dict[int, np.ndarray] = None,
        valid_images: List[str] = None,
    ):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # non_negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        # pose: optional pose as SE(3) matrix
        # positives_poses: optional relative poses of positive examples (ideally refined using ICP)
        # valid_images: valid image files that correspond to the scan
        assert position.shape == (2,)
        assert pose is None or pose.shape == (4, 4)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position
        self.pose = pose
        self.positives_poses = positives_poses
        self.valid_images = valid_images