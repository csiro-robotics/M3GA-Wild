import os 
import pickle 
import numpy as np 
import pandas as pd 
from pathlib import Path
from glob import glob 
from torchpack.utils.config import configs as CFG 
from shapely.geometry import Polygon, Point

from data.dataset import POLY_DICT

class MakeEvalsets:
    def __init__(self):
        self.root_folder = CFG.m3gawild_path
        self.framerate_downsample = CFG.anchor_framerate_downsample
        self.aerial_spatial_downsample = CFG.aerial_spatial_downsample
        
    def get_paths_coords(self, sequence, env):
        # Get images 
        image_paths = sorted(glob(os.path.join(self.root_folder, env, sequence, 'images_shrunk', '*.png')))
        
        # Get coords (only Easting Northing, since z is irrelevant for ground-aerial -- ASSUMES POSES IN WORLD FRAME, NOT CAMERA FRAME)
        df = pd.read_csv(os.path.join(self.root_folder, env, sequence, 'camera_poses_lidar_frame.csv'))
        coords = df[['x','y']].to_numpy()

        # Only keep ground queries from eval regions
        if '_ground' in sequence:
            # Only downsample framerate of ground images (aerial are sampled spatially)
            image_paths = image_paths[::self.framerate_downsample]
            coords = coords[::self.framerate_downsample]
            test_idx = []
            for ii, coord in enumerate(coords):
                point = Point(coord)
                for test_poly in POLY_DICT[env]:
                    if test_poly.contains(point):
                        test_idx.append(ii)
            assert len(test_idx) > 0, 'No test queries found'
            image_paths = [image_paths[ii] for ii in test_idx]
            coords = coords[test_idx]

        # Optionally spatially downsample number of aerial images
        elif '_aerial' in sequence:
            keep_idx = []
            for ii, image_path in enumerate(image_paths):
                # Images are stored as 10000Y.0000X.png, with X and Y corresponding to grid index
                aerial_y_idx, aerial_x_idx = Path(image_path).stem.split('.')
                if int(aerial_y_idx) % self.aerial_spatial_downsample == 0 and int(aerial_x_idx) % self.aerial_spatial_downsample == 0:
                    keep_idx.append(ii)
            image_paths = [image_paths[ii] for ii in keep_idx]
            coords = coords[keep_idx]

        return image_paths, coords
        
    def run(self):
        sequences_venman = [
            '2021_06_11_V-01_ground_sample0.5s',
            '2021_06_11_V-02_ground_sample0.5s',
            '2021_12_16_V-03_ground_sample0.5s',
            '2022_08_12_V-04_ground_sample0.5s',
            '2024_07_09_aerial_sample10m',
        ]
        sequences_karawatha = [
            '2021_06_22_K-01_ground_sample0.5s',
            '2021_06_21_K-02_ground_sample0.5s',
            '2021_12_13_K-03_ground_sample0.5s',
            '2022_08_11_K-04_ground_sample0.5s',
            '2024_07_10_aerial_sample10m',
        ]
       
        info_venman = []
        info_karawatha = []
        
        for idx, seqv in enumerate(sequences_venman):
            paths, coords = self.get_paths_coords(seqv, 'Venman')
            info_seq = {}
            info_seq['name'] = seqv
            info_seq['filenames'] = paths 
            info_seq['coords'] = coords 
            info_seq['timestamps'] = np.float64([os.path.basename(x).strip('.png') for x in paths])
            info_venman.append(info_seq)
            
        for seqk in sequences_karawatha:
            paths, coords = self.get_paths_coords(seqk, 'Karawatha')
            info_seq = {}
            info_seq['name'] = seqk
            info_seq['filenames'] = paths 
            info_seq['coords'] = coords 
            info_seq['timestamps'] = np.float64([os.path.basename(x).strip('.png') for x in paths])
            info_karawatha.append(info_seq) 
            
        venman_eval_info_filepath = os.path.join(os.path.dirname(__file__), 'pickles', f'm3gawild_ground-ds-{self.framerate_downsample}_aerial-ds-{self.aerial_spatial_downsample}_venman_eval_info.pickle')
        karawatha_eval_info_filepath = os.path.join(os.path.dirname(__file__), 'pickles', f'm3gawild_ground-ds-{self.framerate_downsample}_aerial-ds-{self.aerial_spatial_downsample}_karawatha_eval_info.pickle')
        
        os.makedirs(os.path.join(os.path.dirname(__file__), 'pickles'), exist_ok=True)

        with open(venman_eval_info_filepath, 'wb') as f:
            pickle.dump(info_venman, f)
        with open(karawatha_eval_info_filepath, 'wb') as f:
            pickle.dump(info_karawatha, f) 

def make_eval_sets():
    MakeEvalsets().run()
