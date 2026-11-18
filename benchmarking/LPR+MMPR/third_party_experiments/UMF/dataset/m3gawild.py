"""
Dataset wrapper for M3GA-Wild dataset.
"""


import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from typing import Dict, List
from spconv.pytorch.utils import PointToVoxel
import concurrent.futures
from tqdm import tqdm
import open3d as o3d
import dotsi
from dataset.augmentation import Normalize
from dataset.base_datasets import TrainingTuple
from misc.utils import UMFParams
DEBUG = False


class M3GAWildDataset(Dataset):
    """
    Dataset wrapper for M3GA-Wild dataset.
    """
    def __init__(self, dataset_path: str, query_filename: str, image_path: str = None,
                 transform=None, set_transform=None, image_transform=None, use_cloud: bool = True,
                 use_rgb: bool = True, normalize_points: bool = True, cfg: dotsi.Dict = None):
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        print('Loading queries from: {}'.format(self.query_filepath))
        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        self.ground_ndx = {idx for idx, elem in self.queries.items() if '_ground' in elem.rel_scan_filepath}
        self.aerial_ndx = {idx for idx, elem in self.queries.items() if '_aerial' in elem.rel_scan_filepath}
        
        self.image_path = image_path
        self.image_transform = image_transform
        self.image_ext = '.png'
        self.use_cloud = use_cloud
        self.use_rgb = use_rgb
        self.normalize_points = normalize_points
        message = '{} queries in the dataset'.format(len(self))
        message += f' ({len(self.ground_ndx)} ground, {len(self.aerial_ndx)} aerial)'
        print(message)
        self.point_cloud_range = cfg.model.point_cloud.range
        self.cfg = cfg

        self.gen = PointToVoxel(vsize_xyz=cfg.model.point_cloud.voxel_size,
                        coors_range_xyz=cfg.model.point_cloud.range,
                        num_point_features=cfg.model.point_cloud.num_point_features,
                        max_num_voxels=cfg.model.point_cloud.max_num_voxels,
                        max_num_points_per_voxel=cfg.model.point_cloud.max_num_points_per_voxel)
        self.norm_fn = None
        if self.normalize_points:
            self.norm_fn = Normalize(
                    scale_factor=None,
                    unit_sphere_norm=False,
                    zero_mean=True,
                    return_shift_and_scale=False,
                )

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        result = {'ndx': ndx}
        if self.use_cloud:
            # Load point cloud and apply transform
            file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
            query_pc = self.load_pc(file_pathname)
            if self.norm_fn is not None:
                query_pc = self.norm_fn(query_pc)
            if self.transform is not None:
                query_pc = self.transform(query_pc)
            result['cloud'] = query_pc

        if self.use_rgb:
            if self.queries[ndx].valid_images is None:
                raise ValueError('Dataset does not support images')
            random_image_path = np.random.choice(self.queries[ndx].valid_images)
            img = Image.open(os.path.join(self.dataset_path, random_image_path))
            if self.image_transform is not None:
                img = self.image_transform(img)
            result['image'] = img

        return result

    def point_batch_to_voxel(self, clouds):
        device = clouds.device
        # set device cpu
        clouds = clouds.cpu()

        voxel_data = [self.gen(clouds[batch_id]) for batch_id in range(len(clouds))]
        batch_voxel, batch_voxel_coords, batch_num_pts_in_voxels = zip(*voxel_data)
        batch_voxel_coords = list(batch_voxel_coords)


        for batch_id in range(len(batch_voxel_coords)):
            coordinates = batch_voxel_coords[batch_id]

            discrete_coords = torch.cat(
                (
                    torch.zeros(coordinates.shape[0], 1, dtype=torch.int32),
                    coordinates,
                ),
                1,
            )

            batch_voxel_coords[batch_id] = discrete_coords
            # Move batchids to the beginning
            batch_voxel_coords[batch_id][:, 0] = batch_id
        
        batch_voxel_coords = torch.cat(batch_voxel_coords, dim=0)

        feats_batch = torch.ones((batch_voxel_coords.shape[0], 1), dtype=torch.float32, device=device)
        feats_batch = feats_batch.to(device)
        batch_voxel_coords = batch_voxel_coords.to(device)
        return batch_voxel_coords, feats_batch
    

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives

    @staticmethod
    def load_pc(filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        file_ext = os.path.splitext(filename)[-1]
        if file_ext == ".pcd":
            pc_o3d = o3d.io.read_point_cloud(filename)
            pc = np.asarray(pc_o3d.points, dtype=np.float64)
            pc = np.float32(pc)
        elif file_ext == ".bin":
            pc = np.fromfile(filename, dtype=np.float32)
            pc = pc.reshape(-1,4)[:,:3]  # ignoring intensity channel
        else:
            raise NotImplementedError('Invalid point cloud type')
        pc = torch.tensor(pc, dtype=torch.float)

        return pc

class M3GAWildEvalDataset(Dataset):
    """Simple dataset class for evaluation (much faster than original loop)"""
    def __init__(self, set: dict, params: UMFParams, cfg: dotsi.Dict, transform=None, image_transform=None):
        self.set = set
        self.dataset_path = params.dataset_folder
        self.use_cloud = params.use_cloud
        self.use_rgb = params.use_rgb
        self.normalize_points = params.normalize_points
        self.transform = transform
        self.image_transform = image_transform
        self.cfg = cfg
        self.gen = PointToVoxel(
            vsize_xyz=cfg.model.point_cloud.voxel_size,
            coors_range_xyz=cfg.model.point_cloud.range,
            num_point_features=cfg.model.point_cloud.num_point_features,
            max_num_voxels=cfg.model.point_cloud.max_num_voxels,
            max_num_points_per_voxel=cfg.model.point_cloud.max_num_points_per_voxel, 
        )
        self.norm_fn = None
        if self.normalize_points:
            self.norm_fn = Normalize(
                    scale_factor=None,
                    unit_sphere_norm=False,
                    zero_mean=True,
                    return_shift_and_scale=False,
                )

    def __len__(self):
        return len(self.set)

    def __getitem__(self, ndx):
        result = {'ndx': ndx}
        if self.use_cloud:
            # Load point cloud and apply transform
            file_pathname = os.path.join(self.dataset_path, self.set[ndx]['query'])
            query_pc = self.load_pc(file_pathname)
            if self.norm_fn is not None:
                query_pc = self.norm_fn(query_pc)
            if self.transform is not None:
                query_pc = self.transform(query_pc)
            result['cloud'] = query_pc

        if self.use_rgb:
            img = Image.open(os.path.join(self.dataset_path, self.set[ndx]['image']))
            if self.image_transform is not None:
                img = self.image_transform(img)
            result['image'] = img

        return result

    @staticmethod
    def load_pc(filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        file_ext = os.path.splitext(filename)[-1]
        if file_ext == ".pcd":
            pc_o3d = o3d.io.read_point_cloud(filename)
            pc = np.asarray(pc_o3d.points, dtype=np.float64)
            pc = np.float32(pc)
        elif file_ext == ".bin":
            pc = np.fromfile(filename, dtype=np.float32)
            pc = pc.reshape(-1,4)[:,:3]  # ignoring intensity channel
        else:
            raise NotImplementedError('Invalid point cloud type')
        pc = torch.tensor(pc, dtype=torch.float)

        return pc