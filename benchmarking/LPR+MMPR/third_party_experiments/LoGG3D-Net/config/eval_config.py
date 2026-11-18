import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


# Evaluation
eval_arg = add_argument_group('Eval')
eval_arg.add_argument('--eval_pipeline', type=str, default='LOGG3D')
eval_arg.add_argument('--kitti_eval_seq', type=int, default=8)
eval_arg.add_argument('--mulran_eval_seq', type=str,
                      default='Riverside/Riverside_02')
eval_arg.add_argument('--checkpoint_name', type=str,
                      default='/kitti_10cm_loo/2021-09-14_20-28-22_3n24h_Kitti_v10_q29_10s8_263169.pth')
eval_arg.add_argument('--eval_batch_size', type=int, default=1)
eval_arg.add_argument('--test_num_workers', type=int, default=4)
eval_arg.add_argument("--eval_random_rotation", type=str2bool,
                      default=False, help="If random rotation. ")
eval_arg.add_argument("--eval_random_occlusion", type=str2bool,
                      default=False, help="If random occlusion. ")

eval_arg.add_argument("--revisit_criteria", default=3,
                      type=float, help="in meters")
eval_arg.add_argument("--not_revisit_criteria",
                      default=20, type=float, help="in meters")
eval_arg.add_argument("--skip_time", default=30, type=float, help="in seconds")
eval_arg.add_argument("--cd_thresh_min", default=0.001,
                      type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--cd_thresh_max", default=1.0,
                      type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--num_thresholds", default=1000, type=int,
                      help="Number of thresholds. Number of points on PR curve.")

# eval_arg.add_argument("--descriptor_size", default=256,
#                       type=int, help="Global descriptor size. Equals LoGG3D-Net feature_dim**2")
eval_arg.add_argument("--feature_dim", default=16,
                      type=int, help="Feature dim size.")


# Dataset specific configurations
data_arg = add_argument_group('Data')
# KittiDataset #MulRanDataset
data_arg.add_argument('--eval_dataset', type=str, default='KittiDataset')
data_arg.add_argument('--eval_dataset_name', type=str, default='cs_wildplaces')
data_arg.add_argument('--collation_type', type=str,
                      default='default')  # default#sparcify_list
data_arg.add_argument("--eval_save_descriptors", type=str2bool, default=False)
data_arg.add_argument("--eval_save_counts", type=str2bool, default=False)
data_arg.add_argument("--eval_plot_pr_curve", type=str2bool, default=False)
data_arg.add_argument('--num_points', type=int, default=150000)
data_arg.add_argument('--voxel_size', type=float, default=0.05)
data_arg.add_argument("--gp_rem", type=str2bool,
                      default=False, help="Remove ground plane.")
data_arg.add_argument('--eval_feature_distance', type=str,
                      default='cosine')  # cosine#euclidean
data_arg.add_argument("--pnv_preprocessing", type=str2bool,
                      default=False, help="Preprocessing in dataloader for PNV.")

data_arg.add_argument('--kitti_dir', type=str, default='/mnt/088A6CBB8A6CA742/Datasets/Kitti/dataset/',
                      help="Path to the KITTI odometry dataset")
data_arg.add_argument('--kitti_data_split', type=dict, default={
    'train': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'val': [],
    'test': [0]
})

data_arg.add_argument('--mulran_dir', type=str,
                      default='./data/MulRan', help="Path to the MulRan dataset")
data_arg.add_argument("--mulran_normalize_intensity", type=str2bool,
                      default=False, help="Normalize intensity return.")
data_arg.add_argument('--mulran_data_split', type=dict, default={
    'train': ['DCC/DCC_01', 'DCC/DCC_02',
              'Riverside/Riverside_01', 'Riverside/Riverside_03'],
    'val': [],
    'test': ['KAIST/KAIST_01']
})

data_arg.add_argument('--wildplaces_dir', type=str,
                      default='./data/wild_places/data',
                      help="Path to the Wild-Places dataset")

data_arg.add_argument('--cs_wildplaces_dir', type=str,
                      default='./data/CS-Wild-Places/data/CS-Wild-Places/postproc_voxel_0.80m_rmground',
                      help="Path to the CS-Wild-Places dataset")

data_arg.add_argument('--cs_campus3d_dir', type=str,
                      default='./data/cs_campus3d/benchmark_datasets/',
                      help="Path to the CS-Campus3D dataset")

# SGV Eval script configs
data_arg.add_argument('--radius', type=float, nargs='+', default=[5, 20], help='True Positive thresholds in meters')
data_arg.add_argument('--n_samples', type=int, default=None, help='Number of elements sampled from the query sequence')
data_arg.add_argument('--d_thresh', type=float, default=0.4, help='Dist thresholds in meters for sgv')
data_arg.add_argument('--n_topk', type=int, default=20, help='Num retrievals to consider per query')
data_arg.add_argument('--icp_refine', default=True, action='store_true')
data_arg.add_argument('--local_max_eval_threshold', type=float, default=1e10,
                        help='Maxmimum allowable retrieval distance to continue with metric localisation evaluation')
data_arg.add_argument('--sequence_idx', type=int, nargs='+', default=None, help='Index of sequences to consider in eval (relative to dictionaries in `evaluate_wildplaces_utils.py`)')
data_arg.add_argument('--only_global', type=str2bool, default=False, help='Only run global descriptor evaluation')

# Data loader configs
data_arg.add_argument('--train_phase', type=str, default="train")
data_arg.add_argument('--val_phase', type=str, default="val")
data_arg.add_argument('--test_phase', type=str, default="test")
data_arg.add_argument('--use_random_rotation', type=str2bool, default=False)
data_arg.add_argument('--rotation_range', type=float, default=360)
data_arg.add_argument('--use_random_occlusion', type=str2bool, default=False)
data_arg.add_argument('--occlusion_angle', type=float, default=30)
data_arg.add_argument('--use_random_scale', type=str2bool, default=False)
data_arg.add_argument('--min_scale', type=float, default=0.8)
data_arg.add_argument('--max_scale', type=float, default=1.2)


def get_config_eval():
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cfg = get_config_eval()
    dconfig = vars(cfg)
    print(dconfig)
