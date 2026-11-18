#!/bin/bash
pixi run python evaluation/sgv_evaluation/eval_logg3d_sgv.py \
	--eval_pipeline LOGG3D \
	--eval_dataset GeneralDataset \
	--eval_dataset_name m3gawild_airborne \
	--cs_wildplaces_dir ../../../../data/M3GA-Wild \
	--checkpoint_name "checkpoints/logg3dnet_m3ga-wild_airborne.pth" \
	--gp_rem False \
	--feature_dim 32 \
	--voxel_size 0.50 \
	--radius 10 30 \
	--d_thresh 0.4 \
	--local_max_eval_threshold 30 \
	--num_points 150000 \
	--icp_refine \
	--only_global True
