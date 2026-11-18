"""
Script to transform WildCross submap_poses.csv files into georeferenced frame, and 
post-process submaps suitable for training/testing. Also places images in correct subdirectory.

By Ethan Griffiths (Data61, Pullenvale)
"""
import argparse
import shutil
import yaml
import os
from glob import glob

import numpy as np
import pandas as pd
import tqdm

from dataset.CSWildPlaces.processing_utils import (
    CLOUD_SAVE_DIR,
    quaternion_to_rot,
    rot_to_quaternion,
    multiprocessing_func,
    write_bin_pcl,
    load_bin_pcl,
    random_down_sample,
    pnvlad_down_sample,
    voxel_down_sample,
    normalise_pcl,
    remove_ground_CSF,
    get_extrinsic_transform,
    transform_image_trajectory,
)

RANDOM_SEED = 42
IMAGE_DIR = 'images'
IMAGE_POSES = 'camera_poses.csv'
IMAGE_POSES_SAVE_FILE = 'camera_poses_lidar_frame.csv'
IMAGE_DOWNSAMPLED_DIR = 'images_shrunk'

FOREST_SPLIT_DICT = {
    "Karawatha": ['K-01','K-02','K-03','K-04'],
    "Venman": ['V-01','V-02','V-03','V-04']
}

UTM_TF_DICT = {
    "K-01": np.array(
        [
            [ 1.00000006e+00, -6.59700100e-04, -3.27307951e-04,  5.07168741e+05],
            [ 6.59667772e-04,  1.00000003e+00, -1.16085538e-04,  6.94265152e+06],
            [ 3.27392942e-04,  1.15872331e-04,  1.00000002e+00, -6.55034359e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
        ]
    ),
    "K-02": np.array(
        [
            [1.00000006, -0.00065971, -0.00032731, 507168.7409],
            [0.00065967, 1.00000004, -0.00011609, 6942651.521],
            [0.00032739, 0.00011587, 1.00000002, -6.55034349],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "K-03": np.array(
        [
            [ 1.00000013e+00, -6.59710035e-04, -3.27310035e-04,  5.07168741e+05],
            [ 6.59670041e-04,  9.99999991e-01, -1.16089984e-04,  6.94265152e+06],
            [ 3.27390009e-04,  1.15870034e-04,  1.00000001e+00, -6.55034337e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
        ]
    ),
    "K-04": np.array(
        [
            [ 1.00000015e+00, -6.59710912e-04, -3.27309975e-04,  5.07168741e+05],
            [ 6.59671055e-04,  1.00000001e+00, -1.16090001e-04,  6.94265152e+06],
            [ 3.27390009e-04,  1.15870033e-04,  1.00000002e+00, -6.55034347e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
        ]
    ),
    "V-01": np.array(
        [
            [ 9.99986983e-01, -5.10708236e-03, -2.45935476e-04,  5.19798988e+05],
            [ 5.10690254e-03,  9.99986792e-01, -7.37046339e-04,  6.94373697e+06],
            [ 2.49696726e-04,  7.35779681e-04,  9.99999652e-01, -2.35413905e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
        ]
    ),
    "V-02": np.array(
        [
            [ 9.99986989e-01, -5.10708751e-03, -2.45928055e-04,  5.19798988e+05],
            [ 5.10690950e-03,  9.99986787e-01, -7.37040268e-04,  6.94373697e+06],
            [ 2.49688885e-04,  7.35785467e-04,  9.99999644e-01, -2.35413849e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
        ]
    ),
    "V-03": np.array(
        [
            [0.99998626, -0.00525094, -0.00021097, 519799.4990],
            [0.0052508, 0.99998614, -0.00062337, 6943736.910],
            [0.00021424, 0.00062226, 0.99999974, -2.40304419],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "V-04": np.array(
        [
            [ 9.99986110e-01, -5.24609661e-03, -1.98522985e-04,  5.19799213e+05],
            [ 5.24597966e-03,  9.99985976e-01, -6.35054983e-04,  6.94373734e+06],
            [ 2.01853918e-04,  6.34018021e-04,  9.99999732e-01, -2.51588815e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
        ]
    ),
}

SAVE_DIR_DICT = {
    "K-01": "2021_06_22_K-01_ground_sample0.5s",
    "K-02": "2021_06_21_K-02_ground_sample0.5s",
    "K-03": "2021_12_13_K-03_ground_sample0.5s",
    "K-04": "2022_08_11_K-04_ground_sample0.5s",
    "V-01": "2021_06_11_V-01_ground_sample0.5s",
    "V-02": "2021_06_11_V-02_ground_sample0.5s",
    "V-03": "2021_12_16_V-03_ground_sample0.5s",
    "V-04": "2022_08_12_V-04_ground_sample0.5s",
}

def find_nearest(cloud_timestamps_np, timestamp):
    """
    Find nearest cloud file to given timestamp.
    """
    idx = np.argmin(np.abs(cloud_timestamps_np - timestamp))
    return idx

def transform_pose(row, tf_map_to_utm) -> pd.Series:
    """
    Transform single pose value using given transform.
    """
    # Transform to local coordinates
    tf_local_to_map = quaternion_to_rot(np.array(row[['x','y','z','qx','qy','qz','qw']].astype(float)))
        
    local_to_ground = tf_map_to_utm @ tf_local_to_map
        
    # Save new pose in current row
    xyz, quat = rot_to_quaternion(local_to_ground)
    row_new = pd.Series(row, copy=True)
    row_new.loc[['x','y','z','qx','qy','qz','qw']] = [*xyz, *quat]
        
    return row_new

def postprocess_submap(inputs):
    """
    Post-process an individual submap. Order is: Transform to UTM
    -> trim radius -> ground removal -> downsampling -> normalisation.

    Returns transformed pose on success, otherwise returns None if an error
    occured.
    """
    ii, row = inputs
    
    # transform pose
    cloud_file = CLOUD_FILES[ii]
    timestamp = row['timestamp']
    row_new = transform_pose(row, SPLIT_UTM_TF)
    
    # trim cloud
    pts = load_bin_pcl(os.path.join(CLOUD_PATH_UNFILTERED, cloud_file), intensity=True)
    num_pts_before = len(pts)
    pts = pts[np.linalg.norm(pts[:,:2], axis=1) <= args.radius_max]

    # Post-processing    
    if args.remove_ground:
        pts = remove_ground_CSF(pts)
    num_pts = len(pts)
    pts_final = pts    
    if len(pts_final) < args.min_num_points:    # too few points (probably CSF's fault)
        return timestamp
    if args.downsample:
        # FIXME: BELOW LINE IS BAD LOGIC, CAUSES SHORT-CIRCUIT REGARDLESS OF CSF
        if args.downsample_type != 'voxel' and num_pts < args.downsample_target:    # too few points after CSF
            return timestamp
        if args.downsample_type == 'random':
            pts_downsampled = random_down_sample(
                pts, args.downsample_target, RANDOM_SEED
            )
        elif args.downsample_type == 'voxel':
            pts_downsampled = voxel_down_sample(
                pts, args.voxel_size
            )
        elif args.downsample_type == 'pnvlad':
            pts_downsampled = pnvlad_down_sample(
                pts, args.downsample_target, RANDOM_SEED
            )
        else:
            raise(ValueError("Downsample type not recognised"))
        num_pts_downsampled = len(pts_downsampled)
        assert (
            args.downsample_type == 'voxel' or \
            num_pts_downsampled == args.downsample_target
        ),  f'Cloud has {num_pts_downsampled} points after downsampling'
        pts_final = pts_downsampled
    if args.normalise:
        pts_final = normalise_pcl(
            pts_final, pts, args.downsample_target, RANDOM_SEED
        )
    if len(pts_final) < args.min_num_points:    # too few points (probably CSF's fault)
        return timestamp
    
    num_pts_after = len(pts_final)
    
    write_bin_pcl(os.path.join(CLOUD_SAVE_PATH, cloud_file), pcd=pts_final)
    if args.verbose:
        print(f"{ii}: {num_pts_before:>6} pts -> {num_pts_after:>6} pts", flush=True)
    
    return row_new

def main():
    """
    Read csv file of poses in (timestamp,x,y,z,qx,qy,qz,qw) format, and 
    transform to new coordinate frame using relative_transform.
    """
    global CLOUD_TIMESTAMPS, CLOUD_TIMESTAMPS_NP, CLOUD_FILES, SPLIT_UTM_TF, \
        CLOUD_SAVE_PATH, CLOUD_PATH_UNFILTERED

    # Loop through all Karawatha and Venman splits
    assert all(split in os.listdir(args.root) for split in SAVE_DIR_DICT.keys()), (
        "Invalid path to WildCross"
    )

    for forest, splits in FOREST_SPLIT_DICT.items():
        for split in splits:
            print(f"Processing {split}")
            split_path = os.path.join(args.root, split)
            assert os.path.exists(split_path), "Invalid '--root' path, should point to 'WildCross/data'"
            split_save_dir = SAVE_DIR_DICT[split]
            save_path = os.path.join(args.cswildplaces_save_dir, forest, split_save_dir)
            CLOUD_PATH_UNFILTERED = os.path.join(split_path, args.cloud_dir)
            cloud_traj_unfiltered = os.path.join(split_path, args.cloud_traj)
            SPLIT_UTM_TF = UTM_TF_DICT[split]

            # Set cloud dir correctly
            POSTPROC_SAVE_DIR = CLOUD_SAVE_DIR
            if args.downsample:
                downsample_str = 'rand' if args.downsample_type == 'random' else args.downsample_type
                if args.downsample_type == 'voxel':
                    POSTPROC_SAVE_DIR += f'_{downsample_str}_{args.voxel_size:0.2f}m'
                else:
                    POSTPROC_SAVE_DIR += f'_{downsample_str}_{args.downsample_target}'
            if args.remove_ground:
                POSTPROC_SAVE_DIR += '_rmground'
            if args.normalise:
                POSTPROC_SAVE_DIR += '_normalised'
            
            CLOUD_SAVE_PATH = os.path.join(save_path, POSTPROC_SAVE_DIR)
            poses_save_path = os.path.join(save_path, args.cloud_traj)

            os.makedirs(CLOUD_SAVE_PATH, exist_ok=True)
            
            # Copy images (if not already)
            image_poses_path = os.path.join(split_path, IMAGE_POSES)
            image_poses_save_path = os.path.join(save_path, IMAGE_POSES_SAVE_FILE)
            image_path = os.path.join(split_path, IMAGE_DIR)
            image_save_path = os.path.join(save_path, IMAGE_DIR)
            image_downsampled_path = os.path.join(split_path, IMAGE_DOWNSAMPLED_DIR)
            image_downsampled_save_path = os.path.join(save_path, IMAGE_DOWNSAMPLED_DIR)
            try:
                if args.use_symlinks:
                    os.symlink(image_path, image_save_path, target_is_directory=True)
                else:
                    shutil.copytree(image_path, image_save_path)
            except FileExistsError:
                print(f"[WARNING]: Image dir exists for {split}, ignoring... (have not checked if images actually exist or not)")
            try:
                if args.use_symlinks:
                    os.symlink(image_downsampled_path, image_downsampled_save_path, target_is_directory=True)
                else:
                    shutil.copytree(image_downsampled_path, image_downsampled_save_path)
            except FileExistsError:
                print(f"[WARNING]: Image downsampled dir exists for {split}, ignoring... (have not checked if images actually exist or not)")
            # Need to convert image poses to UTM and save manually
            camera_poses = pd.read_csv(
                image_poses_path,
                dtype={'%time':str},
                index_col = False,
            )
            # Below is to follow original Wild-Places format, which used 'timestamp' instead of '%time'
            camera_poses.rename(columns={'%time': 'timestamp'}, inplace=True)
            camera_poses = camera_poses[['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']]    
            # Load camera extrinsics and convert to LiDAR frame
            camera_config_path = glob(os.path.join(split_path, 'p*yaml'))[0]
            try:
                shutil.copy2(camera_config_path, os.path.join(save_path, os.path.basename(camera_config_path)))
            except (FileExistsError, PermissionError):
                print(f"[WARNING]: Camera parameters exist for {split}, ignoring...")
            camera_config = yaml.safe_load(open(camera_config_path))
            T_B_CAM = get_extrinsic_transform(camera_config)
            # xyz = transform_image_trajectory(camera_poses[['x','y','z']].to_numpy(), config)  # from wildcross

            # Save in UTM coords
            if args.save_poses:
                try:
                    shutil.copy2(image_poses_path, os.path.join(save_path, IMAGE_POSES))
                except (FileExistsError, PermissionError):
                    print(f"[WARNING]: Raw camera poses exist for {split}, ignoring...")
                camera_poses_UTM = camera_poses.copy()
                for idx, row in tqdm.tqdm(camera_poses.iterrows(), desc='Converting camera poses', total=len(camera_poses)):
                    row_new = transform_pose(row, SPLIT_UTM_TF @ np.linalg.inv(T_B_CAM))
                    camera_poses_UTM.loc[idx] = row_new
                camera_poses_UTM.to_csv(image_poses_save_path, index=False)
            
            submap_poses = pd.read_csv(
                cloud_traj_unfiltered,
                dtype={'%time':str},
                index_col = False,
            )
            # Below is to follow original Wild-Places format, which used 'timestamp' instead of '%time'
            submap_poses.rename(columns={'%time': 'timestamp'}, inplace=True)
            submap_poses = submap_poses[['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']]
            CLOUD_FILES = sorted(os.listdir(CLOUD_PATH_UNFILTERED))

            inputs = [[ii, row] for ii, row in submap_poses.iterrows()]
            results = multiprocessing_func(
                postprocess_submap, inputs, num_workers=args.num_workers
            )
            # Collate new poses of successfully saved submaps
            submap_poses_temp = [x for x in results if isinstance(x, pd.Series)]               
            submap_poses_save = pd.DataFrame(submap_poses_temp, columns=submap_poses.columns)
            submap_poses_save = submap_poses_save.sort_values(by=['timestamp'])
            if args.save_poses:
                submap_poses_save.to_csv(poses_save_path, index=False)
            
            # Report removed submaps
            if len(submap_poses_save) < len(results):
                print(f"Dropped: {len(results) - len(submap_poses_save)} submaps")

            print(f'Total images: {len(camera_poses)}')

            assert len(submap_poses_save) == len(os.listdir(CLOUD_SAVE_PATH)), \
                "# of entries in poses file and # saved submaps do not match up"
            
            print(f"{split} Done\n")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required = True,
                        help='Path to wild_places/data')
    parser.add_argument('--cswildplaces_save_dir', type=str, required = True,
                        help='Path to M3GA-Wild, where the post-processed ground traversals will be saved.')
    parser.add_argument('--cloud_traj', type=str, default='submap_poses.csv',
                        help='Trajectory file, relative to root')
    parser.add_argument('--save_poses', action='store_true', 
                        help='Save pose file to cswildplaces_save_dir (will overwrite pose file saved by previous runs of this script)')
    parser.add_argument('--cloud_dir', type=str, default="Clouds",
                        help='Clouds directory, relative to root')
    parser.add_argument('--radius_max', type=float, default=30, 
                        help='Max radius (m) of submaps')
    parser.add_argument('--remove_ground', action='store_true', 
                        help='Remove ground points using CSF')
    parser.add_argument('--min_num_points', type=int, default=4096,
                        help='Minimum number of points to consider as a valid submap. Useful after ground removal, incase the submap was nearly entirely flat.')
    parser.add_argument('--downsample', action='store_true', 
                        help='Dowsample point cloud')
    parser.add_argument('--downsample_target', type=int, default=4096,
                        help='Number of points to downsample to')
    parser.add_argument('--downsample_type', type=str, default='voxel', choices = ['pnvlad', 'random', 'voxel'],
                        help='Downsampling method')
    parser.add_argument('--voxel_size', type=float, default=0.8,
                        help='Voxel size (m), if using voxel downsampling')
    parser.add_argument('--normalise', action='store_true', 
                        help='Use PNVLAD normalisation to [-1, 1] range')
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='Number of workers for multiprocessing')
    parser.add_argument('--use_symlinks', action='store_true',
                        help='Use symlinks to link image directories, rather than copying all images')
    parser.add_argument('--verbose', action='store_true' ,
                        help='Print extra info')
    args = parser.parse_args()
    print(args)
    if not (args.remove_ground or args.downsample or args.normalise):
        print("WARNING: No post-processing options selected, so output submaps "
              "will only be cropped to `radius_max`.")
    if not args.downsample or args.downsample_type == 'voxel':
        args.downsample_target = None
    else:
        assert args.downsample_target >= args.min_num_points, \
            "Cannot downsample to less than minimum allowed point count."
    
    main()