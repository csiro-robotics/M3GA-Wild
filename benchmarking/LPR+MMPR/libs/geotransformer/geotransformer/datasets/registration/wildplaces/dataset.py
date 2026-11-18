import os.path as osp
import random

import open3d as o3d
import numpy as np
import torch.utils.data

from geotransformer.utils.common import load_pickle
from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    get_transform_from_rotation_translation,
    get_rotation_translation_from_transform,
    quaternion_to_rotation,
    relative_pose,
)
from geotransformer.utils.registration import get_correspondences


class WildPlacesPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_min_scale=0.8,
        augmentation_max_scale=1.2,
        augmentation_shift=2.0,
        augmentation_rotation=1.0,
        return_corr_indices=False,
        matching_radius=None,
    ):
        """
        NOTE: Currently just using this dataset for debugging with Wild-Places,
              not for correctly training and evaluating on the dataset.
        """
        super(WildPlacesPairDataset, self).__init__()

        self.dataset_root = dataset_root
        self.subset = subset
        self.point_limit = point_limit

        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.augmentation_min_scale = augmentation_min_scale
        self.augmentation_max_scale = augmentation_max_scale
        self.augmentation_shift = augmentation_shift
        self.augmentation_rotation = augmentation_rotation

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        # self.metadata = load_pickle(osp.join(self.dataset_root, 'metadata', f'{subset}.pkl'))
        self.metadata = []
        self.queries = load_pickle(osp.join(self.dataset_root, f'{subset}_evaluation_query.pickle'))
        self.database = load_pickle(osp.join(self.dataset_root, f'{subset}_evaluation_database.pickle'))
        for seq_id, sequence in enumerate(self.queries):
            for query_id, query in sequence.items():
                # NOTE: In Wild-Places inter-sequence eval, pairs are obtained from
                #       two different sequences, but currently only seq_id is set for
                #       the query pcl. This only really matters for logging.
                query_metadata = {'seq_id': seq_id, 'frame0': query_id,
                                  'pcd0': query['query'].replace("Clouds_downsampled", "Clouds")}
                query_pose = get_transform_from_rotation_translation(
                    rotation=quaternion_to_rotation(query['pose'][3:]),
                    translation=query['pose'][:3]
                )
                # Pick the first positive match for each query (priorisiting K-03, which has reverse revisits)
                for database_id in [2,3,0,1]:
                    if database_id not in query or len(query[database_id]) == 0:
                        continue
                    ref_id = query[database_id][0]
                    ref_pcl = self.database[database_id][ref_id]
                    query_metadata.update({'ref_seq_id': database_id, 'frame1': ref_id,
                                           'pcd1': ref_pcl['query'].replace("Clouds_downsampled", "Clouds")})
                    ref_pose = get_transform_from_rotation_translation(
                        rotation=quaternion_to_rotation(ref_pcl['pose'][3:]),
                        translation=ref_pcl['pose'][:3]
                    )
                    break
                transform = relative_pose(query_pose, ref_pose)
                query_metadata.update({'transform': transform})
                self.metadata.append(query_metadata)

    def _augment_point_cloud(self, ref_points, src_points, transform):
        rotation, translation = get_rotation_translation_from_transform(transform)
        # add gaussian noise
        ref_points = ref_points + (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.augmentation_noise
        src_points = src_points + (np.random.rand(src_points.shape[0], 3) - 0.5) * self.augmentation_noise
        # random rotation
        aug_rotation = random_sample_rotation(self.augmentation_rotation)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
        # random scaling
        scale = random.random()
        scale = self.augmentation_min_scale + (self.augmentation_max_scale - self.augmentation_min_scale) * scale
        ref_points = ref_points * scale
        src_points = src_points * scale
        translation = translation * scale
        # random shift
        ref_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        src_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        ref_points = ref_points + ref_shift
        src_points = src_points + src_shift
        translation = -np.matmul(src_shift[None, :], rotation.T) + translation + ref_shift
        # compose transform from rotation and translation
        transform = get_transform_from_rotation_translation(rotation, translation)
        return ref_points, src_points, transform

    def _load_point_cloud(self, file_name):
        points_o3d = o3d.io.read_point_cloud(file_name)
        points = np.asarray(points_o3d.points, dtype=np.float64).astype(np.float32)
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def __getitem__(self, index):
        data_dict = {}

        metadata = self.metadata[index]
        data_dict['seq_id'] = metadata['seq_id']
        data_dict['ref_seq_id'] = metadata['ref_seq_id']
        data_dict['ref_frame'] = metadata['frame0']
        data_dict['src_frame'] = metadata['frame1']

        ref_points = self._load_point_cloud(osp.join(self.dataset_root, metadata['pcd0']))
        src_points = self._load_point_cloud(osp.join(self.dataset_root, metadata['pcd1']))
        transform = metadata['transform']

        if self.use_augmentation:
            ref_points, src_points, transform = self._augment_point_cloud(ref_points, src_points, transform)

        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)

        return data_dict

    def __len__(self):
        return len(self.metadata)
