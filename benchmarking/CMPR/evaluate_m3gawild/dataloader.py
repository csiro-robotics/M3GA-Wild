import torch 
import numpy as np 
import os 
from glob import glob 
import albumentations as A
import cv2 
import pandas as pd 
from tqdm import tqdm 
import torch.nn.functional as F 
import yaml 
from pathlib import Path
import matplotlib.pyplot as plt
from shapely import Point, Polygon
from utils.rangeimage_utils_wildcross import loadCloudFromBinary, createRangeImageM3GAWild, createHeightMapM3GAWild, normalise_range_image
from utils.model_utils import get_transforms_m3gawild
from utils.poses import gravity_align_pc_with_pose
from utils.torch_utils import to_device
from dataloaders.M3GAWildDatasetRangeView import POLY_DICT, make_collate_fn

class EvalDataset:
    def __init__(self, CFG, seq, env, mode):
        self.CFG = CFG 
        self.range_mode = getattr(CFG, 'range_mode', 'range')
        self.transforms = get_transforms_m3gawild(
            mode='val',
            image_size=CFG.size,
            image_centre_crop=getattr(CFG, 'image_centre_crop', False),
            range_image_size=getattr(CFG, 'range_image_size', 224),
            sat493m_norm=('sat493m' in CFG.trained_image_model_name),
        )
        self.data_path = CFG.data_path 
        self.framerate_downsample = 1
        self.aerial_spatial_downsample = 1
        self.cloud_folder = self.CFG.cloud_folder
        self.image_folder = self.CFG.image_folder
        if self.range_mode not in ('range', 'heightmap', 'lidar'):
            raise ValueError('Invalid range image mode')
        self.mode = mode 
        self._get_paths_coords(seq=seq, env=env)
           
    def _get_paths_coords(self, seq, env):
        if self.mode == 'lidar':
            # Get pointclouds 
            self.filepaths = sorted(glob(os.path.join(self.data_path, env, seq, self.cloud_folder, '*.bin')))
            df = pd.read_csv(
                os.path.join(self.data_path, env, seq, 'submap_poses.csv'),
                dtype = {'timestamp': str, 'x': np.float64, 'y': np.float64, 'z': np.float64,
                         'qx': np.float64, 'qy': np.float64, 'qz': np.float64, 'qw': np.float64}
            )

        elif self.mode == 'image':
            # Get images 
            self.filepaths = sorted(glob(os.path.join(self.data_path, env, seq, self.image_folder, '*.png')))
            df = pd.read_csv(
                os.path.join(self.data_path, env, seq, 'camera_poses_lidar_frame.csv'),
                dtype = {'timestamp': str, 'x': np.float64, 'y': np.float64, 'z': np.float64,
                         'qx': np.float64, 'qy': np.float64, 'qz': np.float64, 'qw': np.float64}
            )

        else:
            raise ValueError(self.mode) 
        # Get coords (only Easting Northing, since z is irrelevant for ground-aerial -- ASSUMES POSES IN WORLD FRAME, NOT CAMERA FRAME)
        self.coords = df[['x','y']].to_numpy()
        self.poses = df[['x','y','z','qx','qy','qz','qw']].to_numpy()

        # Remove any submaps that don't have a corresponding file
        filepaths_times = [Path(x).stem for x in self.filepaths]
        timestamps = df['timestamp'].to_list()
        missing_idx = []
        for idx, timestamp in enumerate(timestamps):
            if timestamp not in filepaths_times:
                missing_idx.append(idx)
        if len(missing_idx) > 0:
            print(f'Missing {len(missing_idx)} samples from {seq}, ignoring them instead')
            self.coords = np.delete(self.coords, missing_idx, axis=0)
            self.poses = np.delete(self.poses, missing_idx, axis=0)

        assert len(self.filepaths) == len(self.coords), seq

        # Only keep ground queries from eval regions
        if '_ground' in seq:
            if self.mode == 'image':
                # Only downsample framerate of ground images (aerial are sampled spatially)
                self.filepaths = self.filepaths[::self.framerate_downsample]
                self.coords = self.coords[::self.framerate_downsample]
                self.poses = self.poses[::self.framerate_downsample]
            test_idx = []
            train_idx = []
            for ii, coord in enumerate(self.coords):
                point = Point(coord)
                is_test = False
                for test_poly in POLY_DICT[env]:
                    if test_poly.contains(point):
                        is_test = True
                        test_idx.append(ii)
                        break
                if not is_test:
                    # NOTE: This will include buffer region, but it's mainly for sanity checking
                    train_idx.append(ii)
            if self.CFG.eval_mode == 'test':
                assert len(test_idx) > 0, 'No test queries found'
                self.filepaths = [self.filepaths[ii] for ii in test_idx]
                self.coords = self.coords[test_idx]
                self.poses = self.poses[test_idx]
            elif self.CFG.eval_mode == 'train':
                assert len(train_idx) > 0, 'No train queries found'
                self.filepaths = [self.filepaths[ii] for ii in train_idx]
                self.coords = self.coords[train_idx]
                self.poses = self.poses[train_idx]
            else:
                raise ValueError()
        # Optionally spatially downsample number of aerial images (by default all images used in db)
        elif '_aerial' in seq:
            keep_idx = []
            for ii, image_path in enumerate(self.filepaths):
                # Images are stored as 10000Y.0000X.png, with X and Y corresponding to grid index
                aerial_y_idx, aerial_x_idx = Path(image_path).stem.split('.')
                if int(aerial_y_idx) % self.aerial_spatial_downsample == 0 and int(aerial_x_idx) % self.aerial_spatial_downsample == 0:
                    keep_idx.append(ii)
            self.filepaths = [self.filepaths[ii] for ii in keep_idx]
            self.coords = self.coords[keep_idx]
            self.poses = self.poses[keep_idx]
        else:
            raise ValueError(f'Invalid sequence: {seq}')

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        batch = {}
        batch['coords'] = torch.tensor(self.coords[idx])
        batch['pose'] = torch.tensor(self.poses[idx])
        if self.mode == 'lidar':
            lidar_points = loadCloudFromBinary(self.filepaths[idx])
            # Gravity align pointcloud to ensure heightmap/rangeimage are not distorted
            if self.CFG.gravity_align:
                lidar_points_tmp, _ = gravity_align_pc_with_pose(torch.tensor(lidar_points[:,:3]), self.poses[idx])
                lidar_points[:,:3] = lidar_points_tmp.numpy()
            
            if self.range_mode == 'range':
                lidar_image = createRangeImageM3GAWild(
                    lidar_points,
                    crop=self.CFG.crop,
                    crop_distance=self.CFG.crop_distance,
                    distance_threshold=self.CFG.distance_threshold,
                    max_range=self.CFG.max_range,
                    source=self.CFG.anchor_source,
                    clip_invalid_depth=self.CFG.clip_invalid_depth,
                    idx=idx,
                )
                lidar_image = self.transforms['range_image'](lidar_image)
                batch['lidar_image'] = lidar_image 
            elif self.range_mode == 'heightmap':
                lidar_image = createHeightMapM3GAWild(
                    lidar_points,
                    max_height=self.CFG.max_range,
                    extent=self.CFG.heightmap_extent,
                    resolution=self.CFG.range_image_size,
                    clip_invalid_depth=self.CFG.clip_invalid_depth,
                    idx=idx,
                )
                lidar_image = self.transforms['range_image'](lidar_image)
                batch['lidar_image'] = lidar_image 
            elif self.range_mode == 'lidar':
                lidar_points = torch.tensor(lidar_points[:,:3])  # drop intensity
                lidar_points = self.transforms['lidar'](lidar_points)
                # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
                mask = torch.all(abs(lidar_points) <= 1.0, dim=1)
                lidar_points = lidar_points[mask]
                # Also ensure this will hold if converting coordinate systems
                if self.CFG.coordinates == 'cylindrical':
                    lidar_points_norm = torch.linalg.norm(lidar_points[:, :2], dim=1)[:, None]
                    mask = torch.all(lidar_points_norm <= 1.0, dim=1)
                    lidar_points = lidar_points[mask]
                octree_points = lidar_points
                batch['octree'] = octree_points
        elif self.mode == 'image':
            if 'heightmap' in self.image_folder:
                image = cv2.imread(self.filepaths[idx], cv2.IMREAD_UNCHANGED)
                image = image.astype(np.float32) / 1000.0  # convert from mm to metres
                image = normalise_range_image(
                    image=image,
                    max_range=self.CFG.max_range,
                    clip_invalid_depth=self.CFG.clip_invalid_depth
                )
            else:
                image = cv2.cvtColor(cv2.imread(self.filepaths[idx]), cv2.COLOR_BGR2RGB)
            image = self.transforms['image'](image)
            batch['camera_image'] = image 
        return batch 

@torch.no_grad()
def get_seq_feats_coords(model, CFG, seq, env, mode, debug=False):
    if debug:
        if mode == 'lidar':
            # Get pointclouds 
            poses_file = 'submap_poses.csv'
        elif mode == 'image':
            # Get images 
            poses_file = 'camera_poses_lidar_frame.csv'
        df = pd.read_csv(
            os.path.join(CFG.data_path, env, seq, poses_file),
            dtype = {'timestamp': str, 'x': np.float64, 'y': np.float64, 'z': np.float64,
                        'qx': np.float64, 'qy': np.float64, 'qz': np.float64, 'qw': np.float64}
        )
        all_feats = F.normalize(torch.randn(len(df), 256), dim=1).numpy()
        all_coords = df[['x','y']].to_numpy()
    else:
        # Get features
        collate_fn = make_collate_fn(CFG=CFG)
        dataset = EvalDataset(CFG, seq, env, mode)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32, 
            num_workers=CFG.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        all_feats = []
        all_coords = []
        if mode == 'image':
            desc = f"Getting Database Features (Seq. {seq})"
        elif mode == 'lidar':
            desc = f"Getting Query Features (Seq. {seq})"
        else:
            raise ValueError()

        for batch in tqdm(loader, desc=desc):
            coords = batch['coords'].numpy()
            # batch = {k:v.cuda() for k,v in batch.items()}
            batch = to_device(batch, device='cuda', construct_octree_neigh=True)
            if mode == 'image':
                feats = model.get_camera_embeddings(batch).cpu().numpy()
            elif mode == 'lidar':
                feats = model.get_lidar_embeddings(batch).cpu().numpy()
            all_feats.append(feats)
            all_coords.append(coords)
        
        all_feats = np.concatenate(all_feats)
        all_coords = np.concatenate(all_coords)

    return {'name': seq, 'feats': all_feats, 'coords': all_coords}

def get_image_lidar_features(model, CFG, environment, debug):
    if environment == 'Venman':
        sequences = [
            '2021_06_11_V-01_ground_sample0.5s',
            '2021_06_11_V-02_ground_sample0.5s',
            '2021_12_16_V-03_ground_sample0.5s',
            '2022_08_12_V-04_ground_sample0.5s',
            '2024_07_09_aerial_sample10m',
        ]
    elif environment == 'Karawatha':
        sequences = [
            '2021_06_22_K-01_ground_sample0.5s',
            '2021_06_21_K-02_ground_sample0.5s',
            '2021_12_13_K-03_ground_sample0.5s',
            '2022_08_11_K-04_ground_sample0.5s',
            '2024_07_10_aerial_sample10m',
        ]
    elif environment == 'Samford':
        sequences = [
            '2023_09_17_ground_sample2s',
            '2023_09_18_aerial_sample10m',
        ]
    elif environment == 'QCAT':
        sequences = [
            '2023_09_07_ground_sample2s',
            '2023_09_08_aerial_sample10m',
        ]

    db_sequence = sequences.pop(-1)  # aerial db is last item
    # TODO: Replace logic here to choose image/lidar depending on anchor_modality. Current using anchor_source is quick fix
    query_info_list = []
    if CFG.anchor_source == '_ground':
        db_info = get_seq_feats_coords(model, CFG, db_sequence, environment, mode='image', debug=debug)
        for seq in sequences:
            query_info_list.append(get_seq_feats_coords(model, CFG, seq, environment, mode='lidar', debug=debug))
    elif CFG.anchor_source == '_aerial':
        db_info = get_seq_feats_coords(model, CFG, db_sequence, environment, mode='lidar', debug=debug)
        for seq in sequences:
            query_info_list.append(get_seq_feats_coords(model, CFG, seq, environment, mode='image', debug=debug))
    else:
        raise NotImplementedError()
    
    return query_info_list, db_info 
