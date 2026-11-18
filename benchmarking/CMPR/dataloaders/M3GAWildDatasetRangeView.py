import cv2
import numpy as np
import torch
from utils.rangeimage_utils_wildcross import loadCloudFromBinary, createRangeImageM3GAWild, createHeightMapM3GAWild, normalise_range_image, visualiseRangeImageWildVPR
from utils.model_utils import get_transforms_m3gawild
from utils.poses import gravity_align_pc_with_pose
from dataloaders.coordinate_utils import CylindricalCoordinates

import os
from pathlib import Path
from glob import glob 
import pickle 
import random 
import pandas as pd 
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point
from tqdm import tqdm 
import matplotlib.pyplot as plt
import ocnn
from ocnn.octree import Points, Octree

VAL_FOLDERS = ['2021_06_22_K-01_ground_sample0.5s', '2021_06_21_K-02_ground_sample0.5s',
               '2024_07_10_aerial_sample10m']   # splits to use for validation during training
BASELINE_SPLITS = ['Karawatha', 'Venman']  # splits in baseline train set

### NEW TEST REGIONS WHICH COVER THE ENTIRE SPLITS
# For QCAT
p1 = Polygon([(490500, 6955000), (490500, 6956000), (491500, 6956000), (491500, 6955000)])
# For Samford
p2 = Polygon([(487000, 6969000), (487000, 6971000), (489000, 6971000), (489000, 6969000)])

# For Karawatha (same as wild places, but transformed to UTM frame)
p6 = Polygon([(507018.60467,6942659.3756), (507468.60473,6942659.6724),
              (507468.74853,6942441.6724), (507018.74850,6942441.3756)])
p7 = Polygon([(506953.20227,6943269.3327), (507094.20227,6943269.4257),
              (507094.33093,6943074.4257), (506953.33090,6943074.3327)])
p8 = Polygon([(506655.41198,6942951.1361), (506655.58551,6942688.1361),
              (506847.58554,6942688.2628), (506847.41204,6942951.2627)])
# For Venman (same as wild places, but transformer to UTM frame)
p9 = Polygon([(519331.85162354,6943652.20440674), (519331.19000244,6943778.20266724),
              (519485.18786621,6943779.01129150), (519494.35580444,6943747.05899048),
              (519607.18621826,6943779.65188599), (519607.84783936,6943653.65362549)])
p10 = Polygon([(519722.31359863,6943565.25347900), (519722.54461670,6943521.25408936),
               (519495.54779053,6943520.06213379), (519495.31674194,6943564.06152344)])
p11 = Polygon([(519737.04788208,6943806.33413696), (519894.04573059,6943807.15850830),
               (519941.41265869,6943737.40628052), (519940.15832520,6943595.39773560),
               (519738.16110229,6943594.33709717)])

# # Exclude Karawatha ground region with high drift
# kara_exclude1 = Polygon([[507460, 6942602], [507930, 6942602], [507930, 6941940], 
#                          [506990, 6941940], [506990, 6942340], [507460, 6942340]])

POLY_DICT = {'QCAT':[p1], 'Samford':[p2], 'Karawatha':[p6,p7,p8], 'Venman':[p9, p10, p11]}
# GROUND_EXCLUDE_DICT = {'QCAT':[], 'Samford':[], 'Karawatha':[kara_exclude1], 'Venman':[]}
###

def create_batch(clouds: list[torch.Tensor], CFG):
    """
    Util function to create batches in correct format from an input list of 
    point clouds.

    Args:
        clouds (list[Tensor]): Sequence of point clouds of shape (N, 3).
        CFG: Training parameters for the model.
    """
    coordinates = getattr(CFG, 'coordinates', 'cartesian')
    if coordinates == 'cartesian':
        quantizer = None
    elif coordinates == 'cylindrical':
        quantizer = CylindricalCoordinates(use_octree=True)
    else:
        raise ValueError('Invalid point coordinates')
    if quantizer is not None:
        clouds = [quantizer(e) for e in clouds]
    octrees = []
    # Convert to ocnn Points object, then create Octree
    for cloud in clouds:
        cloud_points_obj = Points(cloud)
        octree = Octree(CFG.octree_depth, CFG.full_depth)
        octree.build_octree(cloud_points_obj)
        octrees.append(octree)
    octrees_merged = ocnn.octree.merge_octrees(octrees)
    # NOTE: remember to construct the neighbor indices before processing (much faster on GPU)
    # octrees_merged.construct_all_neigh()
    batch = {'octree': octrees_merged}
    return batch

def make_collate_fn_aerial_neg_sampling(CFG, dataset):
    def collate_fn(data_list):
        batch = {}
        elem = data_list[0]
        for key in elem.keys():
            batch[key] = []
        for e in data_list:
            for key, val in e.items():
                batch[key].append(torch.as_tensor(val))
        # Sample random aerial negatives here, and add to batch (will be slow, but just need it to work)
        aerial_negatives_idx = random.sample(dataset.aerial_negatives_info.keys(), k=CFG.batch_size)
        aerial_negatives_data_list = []
        # if CFG.anchor_source == '_ground':
        for idx in aerial_negatives_idx:
            aerial_negatives_data_list.append(dataset._get_aerial_negative(idx))
        # Add aerial negatives to batch
        for e in aerial_negatives_data_list:
            for key, val in e.items():
                batch[key].append(torch.as_tensor(val))
        for key, val in batch.items():
            if key == 'octree':
                batch.update(create_batch(val, CFG))
            else:
                batch[key] = torch.stack(val, dim=0)
        return batch
    return collate_fn

def make_collate_fn(CFG):
    def collate_fn(data_list):
        batch = {}
        elem = data_list[0]
        for key in elem.keys():
            batch[key] = []
        for e in data_list:
            for key, val in e.items():
                batch[key].append(torch.as_tensor(val))
        for key, val in batch.items():
            if key == 'octree':
                batch.update(create_batch(val, CFG))
            else:
                batch[key] = torch.stack(val, dim=0)
        return batch
    return collate_fn

def get_dataloader(mode, CFG):
    transforms = get_transforms_m3gawild(
        mode=mode,
        image_size=CFG.size,
        image_centre_crop=getattr(CFG, 'image_centre_crop', False),
        range_image_size=getattr(CFG, 'range_image_size', 224),
        aug_mode=getattr(CFG, 'aug_mode', 1),
        sat493m_norm=('sat493m' in CFG.trained_image_model_name),
    )
    dataset = CLIPDataset(
        transforms=transforms,
        CFG=CFG,
        mode=mode,
    )
    if getattr(CFG, 'sample_aerial_negatives', False):
        collate_fn = make_collate_fn_aerial_neg_sampling(CFG=CFG, dataset=dataset)
    else:
        collate_fn = make_collate_fn(CFG=CFG)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
        collate_fn=collate_fn,
    )
    return dataloader

class M3GAWildRangeDatasetConstruct:
    def __init__(
        self,
        data_path,
        sequences: dict[str, list],
        split: str = 'train',
        pos_radius: float = 15.0,
        buffer_radius: float = 30.0,
        only_ground_aerial: bool = True,  # Only consider ground-aerial positive pairs, not ground-ground or aerial-aerial
        anchor_source: str = '_ground',  # Which viewpoint to use as queries ('_ground' or '_aerial')
        anchor_modality: str = 'lidar',  # Which modality to use as queries (images have more positives to sample)
        cloud_folder: str = 'clouds',  # Point cloud folder to load from (defaults to full resolution clouds)
        image_folder: str = 'images_shrunk',  # Images folder. Supports depth/height images.
    ):
        self.data_path = data_path
        self.sequences = sequences
        if split not in ['train', 'val']:
            raise ValueError(f'Invalid split: {split}')
        self.split = split
        self.pos_radius = pos_radius
        self.buffer_radius = buffer_radius
        self.only_ground_aerial = only_ground_aerial
        if anchor_source not in ['_ground', '_aerial']:
            raise ValueError(f"Anchor source must be '_ground' or '_aerial'")
        self.anchor_source = anchor_source
        if anchor_modality not in ['lidar', 'image']:
            raise ValueError(f"Anchor modality must be 'lidar' or 'image'")
        self.anchor_modality = anchor_modality
        self.cloud_folder = cloud_folder
        self.image_folder = image_folder
        print(self.sequences)
 
    def check_in_test_set(self,
                          sample_idx: int, 
                          test_polygons: list,
                          ground_exclude_polygons: list,
                          all_images: list,
                          all_xyz: list,
                          test_queries_tree : KDTree = None):
        """
        Return which split sample is in, given polygon containing test queries. Run
        twice, first to get the ground test queries, and second to find all submaps
        in the buffer zone.
        """
        split = 'train'
        easting, northing = all_xyz[sample_idx][:2]
        if '_aerial' in all_images[sample_idx]:
            run_type = 'aerial'
        elif '_ground' in all_images[sample_idx]:
            run_type = 'ground'
        else:
            raise ValueError('Invalid image path')
        point = Point(easting, northing)
        for exclude_poly in ground_exclude_polygons:  # Only exclude from ground submaps
            if exclude_poly.contains(point) and run_type == 'ground':
                split = 'buffer'
                return split
        for test_poly in test_polygons:
            if test_poly.contains(point) and run_type == 'ground':
                split = 'test'
                return split
        if test_queries_tree is not None:
            coord = np.array([easting, northing]).reshape(1, -1)
            num_matches = test_queries_tree.query_radius(coord, self.buffer_radius, count_only=True)
            if num_matches[0] > 0:
                split = 'buffer'
        return split 

    def get_positives_for_anchor(self,
                                 anchor_filepath: list[str],
                                 anchor_xy: np.ndarray,
                                 tree: KDTree,
                                 ground_idx: list[int],
                                 aerial_idx: list[int]):
        # Check if anchor is ground or aerial
        if '_ground' in anchor_filepath:
            anchor_type = 'ground'
        elif '_aerial' in anchor_filepath:
            anchor_type = 'aerial'
        else:
            raise ValueError('Invalid path or ground/aerial indices')

        # Get within include radius but outside of exclude radius 
        pos_idx = tree.query_radius(np.array([anchor_xy]), self.pos_radius)[0]

        # Filter positives to enforce cross-source learning (val set always uses ground-aerial pairs)
        if self.only_ground_aerial or self.split == 'val':
            if anchor_type == 'ground':
                pos_idx = np.intersect1d(pos_idx, aerial_idx)
            elif anchor_type == 'aerial':
                pos_idx = np.intersect1d(pos_idx, ground_idx)
        return pos_idx

    def run(self):
        info = {'info': {}, 'aerial_negatives': {}}
        for env, seqs in self.sequences.items():
            print(f'Processing {env}')
            submap_filepaths = []
            image_filepaths = []
            submap_coords = []
            submap_poses = []
            image_coords = []
            for seq in seqs:
                # NOTE: Camera poses are in same coordinate system as submaps in M3GA-Wild-Places
                df_submaps_seq = pd.read_csv(
                    os.path.join(self.data_path, env, seq, 'submap_poses.csv'),
                    dtype = {'timestamp': str, 'x': np.float64, 'y': np.float64, 'z': np.float64,
                             'qx': np.float64, 'qy': np.float64, 'qz': np.float64, 'qw': np.float64}
                )
                df_images_seq = pd.read_csv(
                    os.path.join(self.data_path, env, seq, 'camera_poses_lidar_frame.csv'),
                    dtype = {'timestamp': str, 'x': np.float64, 'y': np.float64, 'z': np.float64,
                             'qx': np.float64, 'qy': np.float64, 'qz': np.float64, 'qw': np.float64}
                )

                # ts_submaps = df_submaps['timestamp'].to_numpy()
                submap_coords_seq = df_submaps_seq[['x','y']].to_numpy()
                submap_poses_seq = df_submaps_seq[['x','y','z','qx','qy','qz','qw']].to_numpy()

                # ts_images = df_images['timestamp'].to_numpy()
                image_coords_seq = df_images_seq[['x','y']].to_numpy()

                submap_filepaths_seq = sorted(glob(os.path.join(self.data_path, env, seq, self.cloud_folder, '*')))
                image_filepaths_seq = sorted(glob(os.path.join(self.data_path, env, seq, self.image_folder, '*')))

                # Remove any submaps that don't have a corresponding file
                submap_filepaths_times_seq = [Path(x).stem for x in submap_filepaths_seq]
                submap_timestamps_seq = df_submaps_seq['timestamp'].to_list()
                submap_missing_idx = []
                for idx, timestamp in enumerate(submap_timestamps_seq):
                    if timestamp not in submap_filepaths_times_seq:
                        submap_missing_idx.append(idx)
                if len(submap_missing_idx) > 0:
                    print(f'Missing {len(submap_missing_idx)} submaps from {seq}, ignoring them instead')
                    submap_coords_seq = np.delete(submap_coords_seq, submap_missing_idx, axis=0)
                    submap_poses_seq = np.delete(submap_poses_seq, submap_missing_idx, axis=0)

                # Remove any images that don't have a corresponding file
                image_filepaths_times_seq = [Path(x).stem for x in image_filepaths_seq]
                image_timestamps_seq = df_images_seq['timestamp'].to_list()
                image_missing_idx = []
                for idx, timestamp in enumerate(image_timestamps_seq):
                    if timestamp not in image_filepaths_times_seq:
                        image_missing_idx.append(idx)
                if len(image_missing_idx) > 0:
                    print(f'Missing {len(image_missing_idx)} images from {seq}, ignoring them instead')
                    image_coords_seq = np.delete(image_coords_seq, image_missing_idx, axis=0)

                assert len(submap_filepaths_seq) == len(submap_coords_seq), seq
                assert len(image_filepaths_seq) == len(image_coords_seq), seq

                submap_filepaths.extend(submap_filepaths_seq)
                image_filepaths.extend(image_filepaths_seq)
                submap_coords.append(submap_coords_seq)
                submap_poses.append(submap_poses_seq)
                image_coords.append(image_coords_seq)
            submap_coords = np.concatenate(submap_coords, axis=0)
            submap_poses = np.concatenate(submap_poses, axis=0)
            image_coords = np.concatenate(image_coords, axis=0)

            # Combine test queries of both modalities for constructing buffer zone
            all_submap_idx = list(range(len(submap_filepaths)))
            test_submap_positions = []
            for aidx in tqdm(all_submap_idx, desc='Constructing submap test set'):
                query_type = self.check_in_test_set(
                    sample_idx=aidx,
                    test_polygons=POLY_DICT[env],
                    ground_exclude_polygons=[],
                    all_images=submap_filepaths,
                    all_xyz=submap_coords,
                    test_queries_tree=None
                )
                if query_type == 'test':
                    test_submap_positions.append(submap_coords[aidx])
            all_image_idx = list(range(len(image_filepaths)))
            test_image_positions = []
            for aidx in tqdm(all_image_idx, desc='Constructing image test set'):
                query_type = self.check_in_test_set(
                    sample_idx=aidx,
                    test_polygons=POLY_DICT[env],
                    ground_exclude_polygons=[],
                    all_images=image_filepaths,
                    all_xyz=image_coords,
                    test_queries_tree=None
                )
                if query_type == 'test':
                    test_image_positions.append(image_coords[aidx])
            test_submap_positions = np.array(test_submap_positions).reshape(-1, 2)  # ensure correct dim for empty lists
            test_image_positions = np.array(test_image_positions).reshape(-1, 2)  # ensure correct dim for empty lists
            test_combined_tree = KDTree(np.concatenate([test_submap_positions, test_image_positions], axis=0))

            # Only include submaps captured within training regions, excluding buffer zone
            split_submap_idx = []
            for aidx in tqdm(all_submap_idx, desc=f'Constructing submap {self.split} set'):
                query_type = self.check_in_test_set(
                    sample_idx=aidx,
                    test_polygons=POLY_DICT[env],
                    ground_exclude_polygons=[],
                    all_images=submap_filepaths,
                    all_xyz=submap_coords,
                    test_queries_tree=test_combined_tree,
                )
                if self.split == 'train':
                    if query_type == 'train':
                        split_submap_idx.append(aidx)
                elif self.split == 'val':
                    in_val_seq = any([val_folder in submap_filepaths[aidx] for val_folder in VAL_FOLDERS])
                    if in_val_seq:
                        if query_type == 'test' or '_aerial' in submap_filepaths[aidx]:  # All aerial submaps form database in val set
                            split_submap_idx.append(aidx)
            if len(split_submap_idx) == 0:
                print(f'No valid samples found in {env}')
                return {}

            # Only include images captured within training regions, excluding buffer zone
            split_image_idx = []
            for aidx in tqdm(all_image_idx, desc=f'Constructing image {self.split} set'):
                query_type = self.check_in_test_set(
                    sample_idx=aidx,
                    test_polygons=POLY_DICT[env],
                    ground_exclude_polygons=[],
                    all_images=image_filepaths,
                    all_xyz=image_coords,
                    test_queries_tree=test_combined_tree,
                )
                if self.split == 'train':
                    if query_type == 'train':
                        split_image_idx.append(aidx)
                elif self.split == 'val':
                    in_val_seq = any([val_folder in image_filepaths[aidx] for val_folder in VAL_FOLDERS])
                    if in_val_seq:
                        if query_type == 'test' or '_aerial' in image_filepaths[aidx]:  # All aerial images form database in val set
                            split_image_idx.append(aidx)
            if len(split_image_idx) == 0:
                print(f'No valid samples found in {env}')
                return {}

            # Construct train/val set 
            split_submap_filepaths = [submap_filepaths[ii] for ii in split_submap_idx]
            split_submap_coords = [submap_coords[ii] for ii in split_submap_idx]
            split_submap_poses = [submap_poses[ii] for ii in split_submap_idx]
            split_image_filepaths = [image_filepaths[ii] for ii in split_image_idx]
            split_image_coords = [image_coords[ii] for ii in split_image_idx]

            # Get cross-modal positives
            if self.anchor_modality == 'lidar':
                num_processed = 0
                num_skipped = 0
                num_positives = 0
                target_kdtree = KDTree(split_image_coords)
                ground_idx, aerial_idx = [], []
                for ii, filepath in enumerate(split_image_filepaths):
                    if '_ground' in filepath:
                        ground_idx.append(ii)
                    elif '_aerial' in filepath:
                        aerial_idx.append(ii)
                    else:
                        raise ValueError('Invalid image file found (not ground or aerial)')
                if len(ground_idx) == 0:
                    print('WARNING: No ground images found (this is okay if using heightmaps)')
                if len(aerial_idx) == 0:
                    raise ValueError('Invalid filepaths (no aerial images found)')
                aerial_negative_idx_set = set(aerial_idx)
                for anchor_idx in tqdm(np.arange(len(split_submap_filepaths)), desc=f'Calculating positives for {env}'):
                    anchor_filepath = split_submap_filepaths[anchor_idx]
                    if self.anchor_source not in anchor_filepath:
                        continue
                    anchor_coord = split_submap_coords[anchor_idx]
                    anchor_pose = split_submap_poses[anchor_idx]
                    num_processed += 1
                    
                    pos_idx = self.get_positives_for_anchor(
                        anchor_filepath=anchor_filepath,
                        anchor_xy=anchor_coord,
                        tree=target_kdtree,
                        ground_idx=ground_idx,
                        aerial_idx=aerial_idx,
                    )
                    if len(pos_idx) == 0:
                        num_skipped +=1
                        continue

                    # If ground anchor has aerial positives, remove them from aerial negatives
                    if self.anchor_source == '_ground':
                        aerial_negative_idx_set.difference_update(set(pos_idx))
                    # If aerial anchor has positives, remove it from aerial negatives
                    elif self.anchor_source == '_aerial':
                        aerial_negative_idx_set.remove(anchor_idx)

                    num_positives += len(pos_idx)
                    pos_images = [split_image_filepaths[pidx] for pidx in pos_idx]
                    pos_coords = [split_image_coords[pidx] for pidx in pos_idx]

                    info['info'][len(info['info'])] = {
                        'submap_filepath': anchor_filepath,
                        'submap_coord': anchor_coord,
                        'submap_pose': anchor_pose,
                        'image_filepaths': pos_images,
                        'image_coords': pos_coords
                    }
                print(f'Saved {num_processed-num_skipped} anchors + {len(aerial_negative_idx_set)}'
                      f' aerial negatives for {env} with {num_positives/len(split_submap_filepaths):.1f}'
                      f' positives on avg ({num_skipped} skipped with no positives)')

                # Also save list of aerial negatives
                aerial_negative_idx = sorted(list(aerial_negative_idx_set))
                for negative_idx in aerial_negative_idx:
                    if self.anchor_source == '_ground':
                        info['aerial_negatives'][len(info['aerial_negatives'])] = {
                            'image_filepath': split_image_filepaths[negative_idx],
                            'image_coord': split_image_coords[negative_idx],
                        }
                    elif self.anchor_source == '_aerial':
                        info['aerial_negatives'][len(info['aerial_negatives'])] = {
                            'submap_filepath': split_submap_filepaths[negative_idx],
                            'submap_coord': split_submap_coords[negative_idx],
                            'submap_pose': split_submap_poses[negative_idx],
                        }
                
            else:
                # TODO: Need to consider anchor subsampling if using ground images as anchor
                raise NotImplementedError()

        print(f'Total num anchors: {len(info["info"])}')
        return info 

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, transforms: dict, CFG, mode):
        self.transforms = transforms
        self.data_path = CFG.data_path
        self.range_mode = getattr(CFG, 'range_mode', 'range')
        if self.range_mode not in ('range', 'heightmap', 'lidar'):
            raise ValueError('Invalid range image mode')
        self.CFG = CFG
        self.get_info(mode)

    def get_info(self, mode):
        train_pickle_name = os.path.join(
            os.path.dirname(__file__),
            f"m3gawild_{self.CFG.cloud_folder}_range_{self.CFG.image_folder}_pos{self.CFG.pos_radius:.1f}_buf{self.CFG.buffer_radius:.1f}"
            f"{'_onlyga' if self.CFG.only_ground_aerial else ''}_anchor-{self.CFG.anchor_source}-{self.CFG.anchor_modality}_train_info.pickle"
        )
        val_pickle_name = os.path.join(
            os.path.dirname(__file__),
            f"m3gawild_{self.CFG.cloud_folder}_range_{self.CFG.image_folder}_pos{self.CFG.pos_radius:.1f}_buf{self.CFG.buffer_radius:.1f}"
            f"{'_onlyga' if self.CFG.only_ground_aerial else ''}_anchor-{self.CFG.anchor_source}-{self.CFG.anchor_modality}_val_info.pickle"
        )
        if not os.path.exists(train_pickle_name) or not os.path.exists(val_pickle_name):
            train_sequences = {
                'Venman': [
                    '2021_06_11_V-01_ground_sample0.5s',
                    '2021_06_11_V-02_ground_sample0.5s',
                    '2021_12_16_V-03_ground_sample0.5s',
                    '2022_08_12_V-04_ground_sample0.5s',
                    '2024_07_09_aerial_sample10m',
                ],
                'Karawatha': [
                    '2021_06_22_K-01_ground_sample0.5s',
                    '2021_06_21_K-02_ground_sample0.5s',
                    '2021_12_13_K-03_ground_sample0.5s',
                    '2022_08_11_K-04_ground_sample0.5s',
                    '2024_07_10_aerial_sample10m',
                ],
            }
            val_sequences = {
                'Karawatha': VAL_FOLDERS,
            }

            train_info = M3GAWildRangeDatasetConstruct(
                self.data_path,
                train_sequences,
                split='train',
                pos_radius=self.CFG.pos_radius,
                buffer_radius=self.CFG.buffer_radius,
                only_ground_aerial=self.CFG.only_ground_aerial,
                anchor_source=self.CFG.anchor_source,
                anchor_modality=self.CFG.anchor_modality,
                cloud_folder=self.CFG.cloud_folder,
                image_folder=self.CFG.image_folder,
            ).run()
            val_info = M3GAWildRangeDatasetConstruct(
                self.data_path,
                val_sequences,
                split='val',
                pos_radius=self.CFG.pos_radius,
                buffer_radius=self.CFG.buffer_radius,
                only_ground_aerial=self.CFG.only_ground_aerial,
                anchor_source=self.CFG.anchor_source,
                anchor_modality=self.CFG.anchor_modality,
                cloud_folder=self.CFG.cloud_folder,
                image_folder=self.CFG.image_folder,
            ).run()
            with open(train_pickle_name, 'wb') as f:
                pickle.dump(train_info, f)
            with open(val_pickle_name, 'wb') as f:
                pickle.dump(val_info, f)

        if mode == 'train': 
            info_dict = pickle.load(open(train_pickle_name, 'rb'))
        else:
            info_dict = pickle.load(open(val_pickle_name, 'rb'))
        self.info = info_dict['info']
        self.aerial_negatives_info = info_dict['aerial_negatives']

    def __len__(self):
        return len(self.info)

    def flush(self):
        pass 

    def _get_aerial_negative(self, idx):
        """
        Sample an aerial negative
        """
        if self.CFG.anchor_modality != 'lidar':
            raise NotImplementedError()
        info_idx = self.aerial_negatives_info[idx]
        
        if self.CFG.anchor_source == '_ground':
            image_path = info_idx['image_filepath']
            image_coord = info_idx['image_coord']
            image = self._load_image(image_path=image_path)
            return {'camera_image': image, 'camera_coord': image_coord}
        elif self.CFG.anchor_source == '_aerial':
            raise NotImplementedError()
            lidar_path = info_idx['submap_filepath']
            lidar_coord = info_idx['submap_coord']
            lidar_pose = info_idx['submap_pose']

            # TODO
            
            if self.range_mode == 'lidar':
                return {'octree': octree_points, 'lidar_coord': lidar_coord}
            else:
                return {'lidar_image': lidar_image, 'lidar_coord': lidar_coord}

    def _load_image(self, image_path, idx=0):
        # Load image
        if 'heightmap' in self.CFG.image_folder:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            image = image.astype(np.float32) / 1000.0  # convert from mm to metres
            image = normalise_range_image(
                image=image,
                max_range=self.CFG.max_range,
                clip_invalid_depth=self.CFG.clip_invalid_depth
            )
            # load range augmentation if image is a chmv2 heightmap
            image = self.transforms['range_image'](image)
        else:
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            image = self.transforms['image'](image)
        return image

    def __getitem__(self, idx):
        info_idx = self.info[idx]
        lidar_path = info_idx['submap_filepath']
        lidar_coord = info_idx['submap_coord']
        lidar_pose = info_idx['submap_pose']
        
        pos_image_idx = random.choice(range(len(info_idx['image_filepaths'])))
        image_path = info_idx['image_filepaths'][pos_image_idx]
        image_coord = info_idx['image_coords'][pos_image_idx]

        # Load image
        image = self._load_image(image_path=image_path, idx=idx)

        # Load lidar
        lidar_points = loadCloudFromBinary(lidar_path)

        # Gravity align pointcloud to ensure heightmap/rangeimage are not distorted
        if self.CFG.gravity_align:
            lidar_points_tmp, _ = gravity_align_pc_with_pose(torch.tensor(lidar_points[:,:3]), lidar_pose)
            lidar_points[:,:3] = lidar_points_tmp.numpy()
        
        if self.range_mode == 'range':
            lidar_image = createRangeImageM3GAWild(
                lidar_points,
                self.CFG.crop,
                self.CFG.crop_distance,
                self.CFG.distance_threshold,
                max_range=self.CFG.max_range,
                source=self.CFG.anchor_source,
                clip_invalid_depth=self.CFG.clip_invalid_depth,
                idx=idx,
            )
            lidar_image = self.transforms['range_image'](lidar_image)
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

        # ## DEBUGGING ##
        # from utils.rangeimage_utils_wildcross import visualiseRangeImageWildVPR
        # img_mean = np.array((0.485, 0.456, 0.406))
        # img_std = np.array((0.229, 0.224, 0.225))
        # lidar_image_unnorm = (lidar_image.permute(1,2,0) * (img_std * 255.0) + img_mean * 255.0)
        # max_range_tmp = self.CFG.max_range
        # if max_range_tmp is None:
        #     max_range_tmp = 50
        # lidar_image_unnorm = lidar_image_unnorm / (255 / max_range_tmp)
        # plt.imshow(visualiseRangeImageWildVPR(lidar_image_unnorm[:,:,0].numpy(), max_range=max_range_tmp))
        # plt.tight_layout()
        # if self.range_mode == 'range':
        #     plt.savefig(f'range_image_m3ga_debug/{idx:>06}_rangeimage_aug.png')
        #     # plt.savefig(f'tmp_rangeimage_aug.png')
        # elif self.range_mode == 'heightmap':
        #     plt.savefig(f'range_image_m3ga_debug/{idx:>06}_heightmap_aug.png')
        #     # plt.savefig(f'tmp_heightmap_aug.png')
        # plt.close()
        # ###############
    
        if self.range_mode == 'lidar':
            return {'camera_image': image, 'octree': octree_points, 'camera_coord': image_coord, 'lidar_coord': lidar_coord}
        else:
            return {'camera_image': image, 'lidar_image': lidar_image, 'camera_coord': image_coord, 'lidar_coord': lidar_coord}
