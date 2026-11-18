
import os 
import pickle 
import random 

import torch 
import numpy as np 
import PIL.Image as Image 

import pandas as pd 
from tqdm import tqdm 
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point

VAL_FOLDERS = ['2021_06_22_K-01_ground_sample0.5s', '2024_07_10_aerial_sample10m']   # splits to use for validation during training
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


class M3GAWildInfoConstructor:
    def __init__(self,
                 data_root: str, 
                 split: str,
                 img_per_place: int,
                 pos_include_radius: float,
                 pos_exclude_radius: float,
                 fov_include_radius: float,
                 buffer_radius: float,
                 only_ground_aerial: bool,
                 anchor_framerate_downsample: int):
        if split not in ['train', 'val']:
            raise ValueError('Invalid split')
        self.dataset_path = data_root 
        self.split = split
        self.img_per_place = img_per_place
        self.image_folder_name = 'images_shrunk'
        self.pos_include_radius = pos_include_radius 
        self.pos_exclude_radius = pos_exclude_radius 
        self.fov_include_radius = fov_include_radius 
        self.buffer_radius = buffer_radius
        self.only_ground_aerial = only_ground_aerial
        self.anchor_framerate_downsample = anchor_framerate_downsample
        self.EPS = 1e-8
        
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
                                 anchor_idx: int,
                                 tree: KDTree,
                                 all_images: list[str],
                                 all_xy: list[np.ndarray],
                                 all_uvectors: list[np.ndarray],
                                 ground_idx: list[int],
                                 aerial_idx: list[int]):
        # Check if anchor is ground or aerial
        if '_ground' in all_images[anchor_idx]:
            anchor_type = 'ground'
        elif '_aerial' in all_images[anchor_idx]:
            anchor_type = 'aerial'
        else:
            raise ValueError('Invalid path or ground/aerial indices')
                                 
        # Check to see if a frame can see a corresponding place 
        anchor_xy = all_xy[anchor_idx]
        anchor_uvector = all_uvectors[anchor_idx]

        # Get within include radius but outside of exclude radius 
        pos_idx = tree.query_radius(np.array([anchor_xy]), self.pos_include_radius)[0]
        exclude = tree.query_radius(np.array([anchor_xy]), self.pos_exclude_radius)[0]
        pos_idx = np.setdiff1d(pos_idx, exclude)

        # Filter positives to enforce cross-source learning (val set always uses ground-aerial pairs)
        if self.only_ground_aerial or self.split == 'val':
            if anchor_type == 'ground':
                pos_idx = np.intersect1d(pos_idx, aerial_idx)
            elif anchor_type == 'aerial':
                pos_idx = np.intersect1d(pos_idx, ground_idx)

        # Exit early if no positives found
        if len(pos_idx) == 0:
            return None

        # Get images, xyz, uvectors 
        pos_images = [all_images[idx] for idx in pos_idx]
        pos_xy = [all_xy[idx] for idx in pos_idx]
        pos_uvectors = np.array([all_uvectors[idx] for idx in pos_idx])

        # Apply FoV threshold to ground-ground training pairs
        if anchor_type == 'ground' and self.split == 'train':
            pos_aerial_idx = np.array([ii for ii, filepath in enumerate(pos_images) if '_aerial' in filepath], dtype=np.int64)
            angle = np.arccos(np.clip(np.einsum('HD,HD->H', anchor_uvector.reshape(1,2), pos_uvectors), -1.0, 1.0)) * 180 / np.pi 
            keep_idx = np.nonzero(angle <= self.fov_include_radius)[0]
            # Angle is invalid for aerial images, so keep all aerial positives
            keep_idx = np.union1d(keep_idx, pos_aerial_idx)
            filtered_pos_images = [pos_images[x] for x in keep_idx]
            filtered_pos_xy = [pos_xy[x] for x in keep_idx] 
        else:
            filtered_pos_images = pos_images
            filtered_pos_xy = pos_xy

        # Add anchor to first element of positives
        filtered_pos_images.insert(0, all_images[anchor_idx])
        filtered_pos_xy.insert(0, anchor_xy)

        if len(filtered_pos_images) < self.img_per_place:
            return None 
        else:
            return {'images': filtered_pos_images, 'coords': filtered_pos_xy}
        
    def construct_split_m3gawild_environment(self, environment):
        if environment.lower() == 'venman':
            sequences = [
                '2021_06_11_V-01_ground_sample0.5s',
                '2021_06_11_V-02_ground_sample0.5s',
                '2021_12_16_V-03_ground_sample0.5s',
                '2022_08_12_V-04_ground_sample0.5s',
                '2024_07_09_aerial_sample10m',
            ]
        elif environment.lower() == 'karawatha':
            sequences = [
                '2021_06_22_K-01_ground_sample0.5s',
                '2021_06_21_K-02_ground_sample0.5s',
                '2021_12_13_K-03_ground_sample0.5s',
                '2022_08_11_K-04_ground_sample0.5s',
                '2024_07_10_aerial_sample10m',
            ]
        print(f'{self.split} sequences for environment {environment}: {sequences}')

        # Get timestamp, positions and unit vectors for all images
        # We will use this to determine if an image belongs to a place 
        filepaths = []
        positions = []
        unit_vectors = []
        for seq in tqdm(sequences, desc = 'Reading Sequences'):
            df = pd.read_csv(os.path.join(self.dataset_path, environment, seq, 'camera_poses_lidar_frame.csv'),
                             dtype = {'timestamp': str, 'x': np.float64, 'y': np.float64, 'z': np.float64,
                                      'qx': np.float64, 'qy': np.float64, 'qz': np.float64, 'qw': np.float64})

            filepaths_seq = [os.path.join(self.dataset_path, environment, seq, self.image_folder_name, f"{ts}.png") for ts in df['timestamp']]
            # NOTE: ONLY USING XY POSITIONS FOR GROUND-AERIAL MATCHING. ASSUMING POSES ARE IN WORLD FRAME, NOT CAMERA FRAME.
            positions_seq = df[['x','y']].to_numpy()
            unit_vectors_seq = (positions_seq[1:] - positions_seq[:-1]) / (np.linalg.norm(positions_seq[1:] - positions_seq[:-1], axis=1, keepdims=True) + self.EPS)
            unit_vectors_seq = np.concatenate([unit_vectors_seq, unit_vectors_seq[-1].reshape(1,-1)], axis = 0 )

            filepaths += filepaths_seq
            positions += list(positions_seq)
            unit_vectors += list(unit_vectors_seq)

        # Only include images captured within training regions, excluding buffer zone
        all_idx = list(range(len(filepaths)))
        test_positions = []
        for aidx in tqdm(all_idx, desc='Constructing test set'):
            query_type = self.check_in_test_set(
                sample_idx=aidx,
                test_polygons=POLY_DICT[environment],
                ground_exclude_polygons=[],
                all_images=filepaths,
                all_xyz=positions,
                test_queries_tree=None
            )
            if query_type == 'test':
                test_positions.append(positions[aidx])
        test_tree = KDTree(np.array(test_positions))
        split_idx = []
        for aidx in tqdm(all_idx, desc=f'Constructing {self.split} set'):
            query_type = self.check_in_test_set(
                sample_idx=aidx,
                test_polygons=POLY_DICT[environment],
                ground_exclude_polygons=[],
                all_images=filepaths,
                all_xyz=positions,
                test_queries_tree=test_tree,
            )
            if self.split == 'train':
                if query_type == 'train':
                    split_idx.append(aidx)
            elif self.split == 'val':
                in_val_seq = any([val_folder in filepaths[aidx] for val_folder in VAL_FOLDERS])
                if in_val_seq:
                    if query_type == 'test' or '_aerial' in filepaths[aidx]:  # All aerial images form database in val set
                        split_idx.append(aidx)
        if len(split_idx) == 0:
            print(f'No valid samples found in {environment}')
            return []

        # Construct train/val set 
        split_filepaths = [filepaths[ii] for ii in split_idx]
        split_positions = [positions[ii] for ii in split_idx]
        split_unit_vectors = [unit_vectors[ii] for ii in split_idx]
        ground_idx, aerial_idx = [], []
        for ii, filepath in enumerate(split_filepaths):
            if '_ground' in filepath:
                ground_idx.append(ii)
            elif '_aerial' in filepath:
                aerial_idx.append(ii)
            else:
                raise ValueError('Invalid image file found (not ground or aerial)')
        if len(ground_idx) == 0:
            raise ValueError('Invalid filepaths (no ground images found)')
        if len(aerial_idx) == 0:
            raise ValueError('Invalid filepaths (no aerial images found)')
        # Only apply framerate downsampling to ground images (add all aerial images as potential anchors for training)
        anchor_idx = set(list(range(len(split_filepaths)))[::self.anchor_framerate_downsample])
        if self.split == 'train':
            anchor_idx = sorted(set.union(anchor_idx, set(aerial_idx)))
        elif self.split == 'val':
            anchor_idx = sorted(set.difference(anchor_idx, set(aerial_idx)))

        split_tree = KDTree(np.array(split_positions))
        info = []
        ground_count = 0
        aerial_count = 0
        for aidx in tqdm(anchor_idx, desc = 'Getting positives for each training idx'):
            info_idx = self.get_positives_for_anchor(
                aidx,
                split_tree,
                split_filepaths,
                split_positions,
                split_unit_vectors,
                ground_idx,
                aerial_idx
            )
            if info_idx is not None:
                info.append(info_idx)
            else:
                if aidx in aerial_idx:
                    aerial_count += 1
                else:
                    ground_count += 1
        print(f'Total of {len(info)} {self.split} samples available ({ground_count} ground + {aerial_count} aerial excluded due to not having enough potential positives)')
        return info 

    def run(self):
        info_venman = self.construct_split_m3gawild_environment('Venman')
        info_karawatha = self.construct_split_m3gawild_environment('Karawatha')
        
        info_combined = info_venman + info_karawatha

        if len(info_combined) == 0:
            raise RuntimeError(f'0 samples found for {self.split} split, check that pos_include_radius and img_per_place are sensible for the dataset')
        
        print(f'Total of {len(info_combined)} {self.split} samples ')
        print(f"Average of {np.mean([len(x['images']) for x in info_combined]):.1f} positives per training sample")
        return info_combined


class M3GAWildDataset:
    def __init__(self,
                data_root: str,
                split: str,
                pos_exclude_radius: float,
                pos_include_radius: float,
                fov_include_radius: float,
                buffer_radius: float,
                only_ground_aerial: bool,
                anchor_framerate_downsample: int,
                img_per_place: int,
                transform):
        if split not in ['train', 'val']:
            raise ValueError('Invalid split')
        self.data_root = data_root
        self.split = split
        self.pos_exclude_radius = pos_exclude_radius
        self.pos_include_radius = pos_include_radius
        self.fov_include_radius = fov_include_radius
        self.buffer_radius = buffer_radius
        self.only_ground_aerial = only_ground_aerial
        self.anchor_framerate_downsample = anchor_framerate_downsample
        self.img_per_place = img_per_place
        self.transform = transform
        
        self.info = self.get_info()
        
    def get_info(self):
        only_ga_str = '_onlyga' if self.only_ground_aerial else ''
        info_pickle_path = os.path.join(os.path.dirname(__file__), 'pickles_m3gawild', f'{self.split}_info_pir{self.pos_include_radius}_per_{self.pos_exclude_radius}_fovr_{self.fov_include_radius}_bufr_{self.buffer_radius}_afds_{self.anchor_framerate_downsample}{only_ga_str}.pickle')
        if os.path.exists(info_pickle_path):
            print(f'Loading {self.split} info from: {info_pickle_path}')
            info = pickle.load(open(info_pickle_path, 'rb'))
        else:
            os.makedirs(os.path.dirname(info_pickle_path), exist_ok=True)
            info = M3GAWildInfoConstructor(
                self.data_root, self.split, self.img_per_place,
                pos_include_radius=self.pos_include_radius,
                pos_exclude_radius=self.pos_exclude_radius,
                fov_include_radius=self.fov_include_radius,
                buffer_radius=self.buffer_radius,
                only_ground_aerial=self.only_ground_aerial,
                anchor_framerate_downsample=self.anchor_framerate_downsample).run()
            with open(info_pickle_path, 'wb') as f:
                pickle.dump(info, f)
        return info 
    
    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        info_idx = self.info[idx]
        pos_image_paths = info_idx['images']
        pos_coords = info_idx['coords']

        # Load anchor (always first element of list)
        select_idx = [0]
        # Randomly select positives
        select_idx += random.sample(list(range(1, len(pos_image_paths))), self.img_per_place-1)
        select_images = []
        select_coords = []
        
        for s_idx in select_idx:
            im = Image.open(os.path.join(self.data_root, 'images', pos_image_paths[s_idx]))
            im = self.transform(im)
            select_images.append(im)
            select_coords.append(pos_coords[s_idx])
            
        select_images = torch.tensor(np.stack(select_images))
        select_coords = torch.tensor(np.stack(select_coords))
        
        return select_images, torch.tensor(idx).repeat(self.img_per_place), select_coords


class WildCrossInfoConstructor:
    def __init__(self,
                 data_root, 
                 img_per_place,
                 pos_include_radius,
                 pos_exclude_radius,
                 fov_include_radius,
                 anchor_framerate_downsample,
                 split_idx):
        self.dataset_path = data_root 
        self.img_per_place = img_per_place
        self.image_folder_name = 'images_shrunk'
        self.pos_include_radius = pos_include_radius 
        self.pos_exclude_radius = pos_exclude_radius 
        self.fov_include_radius = fov_include_radius 
        self.anchor_framerate_downsample = anchor_framerate_downsample
        self.split_idx = split_idx
        
    def get_positives_for_anchor(self, anchor_idx, tree, all_images, all_xyz, all_uvectors):
        # Check to see if a frame can see a corresponding place 
        anchor_xyz = all_xyz[anchor_idx]
        anchor_uvector = all_uvectors[anchor_idx]

        # Get within include radius but outside of exclude radius 
        pos_idx = tree.query_radius(np.array([anchor_xyz]), self.pos_include_radius)[0]
        exclude = tree.query_radius(np.array([anchor_xyz]), self.pos_exclude_radius)[0]
        pos_idx = [x for x in pos_idx if x not in exclude]

        # Get images, xyz, uvectors 
        pos_images = [all_images[idx] for idx in pos_idx]
        pos_xyz = np.array([all_xyz[idx] for idx in pos_idx])
        pos_uvectors = np.array([all_uvectors[idx] for idx in pos_idx])

        angle = np.arccos(np.clip(np.einsum('HD,HD->H', anchor_uvector.reshape(1,3), pos_uvectors), -1.0, 1.0)) * 180 / np.pi 
        keep_idx = np.nonzero(angle <= self.fov_include_radius)[0]

        filtered_pos_images = [pos_images[x] for x in keep_idx]
        filtered_pos_xyz = [pos_xyz[x] for x in keep_idx] 


        if len(filtered_pos_images) < self.img_per_place:
            return None 
        else:
            return {'images': filtered_pos_images, 'coords': filtered_pos_xyz}
        
    def construct_train_set_wildcross_environment(self, environment):
        if environment == 'venman':
            sequences = ['V-01', 'V-02', 'V-03', 'V-04']
        elif environment == 'karawatha':
            sequences = ['K-01', 'K-02', 'K-03', 'K-04']
        # NOTE split idx should follow convention of datasets 
        # (e.g. split_idx 1 refers to K-01 and V-01)
        sequences.pop(self.split_idx-1)
        print(f"Training Sequences for environment {environment}: {sequences}")
        
        # Get timestamp, positions and unit vectors for all images
        # We will use this to determine if an image belongs to a place 
        filepaths = []
        positions = []
        unit_vectors = []
        for seq in tqdm(sequences, desc = "Reading Sequences"):
            df = pd.read_csv(os.path.join(self.dataset_path, seq, 'camera_poses.csv'),
                             dtype = {'%time': str, 'x': np.float64, 'y': np.float64, 'z': np.float64,
                                      'qx': np.float64, 'qy': np.float64, 'qz': np.float64, 'qw': np.float64})
            
            if environment == 'karawatha':
                df['y'] += 10000 # Offset to spatially separate the two environments 

            filepaths_seq = [os.path.join(self.dataset_path, seq, self.image_folder_name, f"{ts}.png") for ts in df['%time']]
            positions_seq = df[['x','y','z']].to_numpy()
            unit_vectors_seq = (positions_seq[1:] - positions_seq[:-1]) / np.linalg.norm(positions_seq[1:] - positions_seq[:-1], axis=1, keepdims=True)
            unit_vectors_seq = np.concatenate([unit_vectors_seq, unit_vectors_seq[-1].reshape(1,-1)], axis = 0 )

            filepaths += filepaths_seq
            positions += list(positions_seq)
            unit_vectors += list(unit_vectors_seq)
        
        # Construct training set 
        anchor_idx = list(range(len(filepaths)))[::self.anchor_framerate_downsample]
        tree = KDTree(np.array(positions))
        info = []
        for aidx in tqdm(anchor_idx, desc = "Getting positives for each training idx"):
            info_idx = self.get_positives_for_anchor(aidx, tree, filepaths, positions, unit_vectors)
            if info_idx is not None:
                # print(type(info))
                info.append(info_idx)
        print(f"Total of {len(info)} Training Samples available ({len(anchor_idx) - len(info)} Excluded due to not having enough potential positives)")
        return info 
        
    def run(self):
        info_venman = self.construct_train_set_wildcross_environment('venman')
        info_karawatha = self.construct_train_set_wildcross_environment('karawatha')
        
        info_combined = info_venman + info_karawatha
        
        print(f"Total of {len(info_combined)} Training Samples ")
        print(f"Average of {np.mean([len(x['images']) for x in info_combined])} positives per training sample")
        return info_combined

class WildCrossDataset:
    def __init__(self,
                data_root,
                pos_exclude_radius,
                pos_include_radius,
                fov_include_radius,
                anchor_framerate_downsample,
                img_per_place,
                transform,
                split_idx):
        
        self.data_root = data_root
        self.pos_exclude_radius = pos_exclude_radius
        self.pos_include_radius = pos_include_radius
        self.pos_include_radius = pos_include_radius
        self.fov_include_radius = fov_include_radius
        self.anchor_framerate_downsample = anchor_framerate_downsample
        self.img_per_place = img_per_place
        self.img_per_place = img_per_place
        self.transform = transform
        self.split_idx = split_idx
        
        self.info = self.get_info()
        
    def get_info(self):
        info_pickle_path = os.path.join(os.path.dirname(__file__), 'pickles', f"train_info_pir{self.pos_include_radius}_per_{self.pos_exclude_radius}_fovr_{self.fov_include_radius}_afds_{self.anchor_framerate_downsample}_sidx_{self.split_idx}.pickle")
        if os.path.exists(info_pickle_path):
            info = pickle.load(open(info_pickle_path, 'rb'))
        else:
            os.makedirs(os.path.dirname(info_pickle_path), exist_ok=True)
            info = WildCrossInfoConstructor(
                self.data_root, self.img_per_place,
                pos_include_radius=self.pos_include_radius,
                pos_exclude_radius=self.pos_exclude_radius,
                fov_include_radius=self.fov_include_radius,
                anchor_framerate_downsample=self.anchor_framerate_downsample,
                split_idx=self.split_idx).run()
            with open(info_pickle_path, 'wb') as f:
                pickle.dump(info, f)
        return info 
    
    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        info_idx = self.info[idx]
        pos_image_paths = info_idx['images']
        pos_coords = info_idx['coords']
        
        select_idx = random.sample(list(range(len(pos_image_paths))), self.img_per_place)
        select_images = []
        select_coords = []
        
        for s_idx in select_idx:
            im = Image.open(os.path.join(self.data_root, 'images', pos_image_paths[s_idx]))
            im = self.transform(im)
            select_images.append(im)
            select_coords.append(pos_coords[s_idx])
            
        select_images = torch.tensor(np.stack(select_images))
        select_coords = torch.tensor(np.stack(select_coords))
        
        return select_images, torch.tensor(idx).repeat(self.img_per_place), select_coords
