#!/bin/bash

python evaluation/sgv_evaluation/eval_logg3d_sgv.py \
	--eval_pipeline LOGG3D \
	--eval_dataset GeneralDataset \
	--eval_dataset_name cs_wildplaces \
	--checkpoint_name checkpoints/logg3dnet_cs-wild-places.pth \
	--gp_rem False \
	--feature_dim 32 \
	--voxel_size 0.50 \
	--radius 10 30 \
	--d_thresh 0.4 \
	--local_max_eval_threshold 30 \
	--num_points 150000 \
	--icp_refine \
	--sequence_idx 0 \
	&> ./results_cs_wildplaces_sgv_metloc_success_kara.txt &

python evaluation/sgv_evaluation/eval_logg3d_sgv.py \
	--eval_pipeline LOGG3D \
	--eval_dataset GeneralDataset \
	--eval_dataset_name cs_wildplaces \
	--checkpoint_name checkpoints/logg3dnet_cs-wild-places.pth \
	--gp_rem False \
	--feature_dim 32 \
	--voxel_size 0.50 \
	--radius 10 30 \
	--d_thresh 0.4 \
	--local_max_eval_threshold 30 \
	--num_points 150000 \
	--icp_refine \
	--sequence_idx 1 \
	&> ./results_cs_wildplaces_sgv_metloc_success_venman.txt &

python evaluation/sgv_evaluation/eval_logg3d_sgv.py \
	--eval_pipeline LOGG3D \
	--eval_dataset GeneralDataset \
	--eval_dataset_name cs_wildplaces \
	--checkpoint_name checkpoints/logg3dnet_cs-wild-places.pth \
	--gp_rem False \
	--feature_dim 32 \
	--voxel_size 0.50 \
	--radius 10 30 \
	--d_thresh 0.4 \
	--local_max_eval_threshold 30 \
	--num_points 150000 \
	--icp_refine \
	--sequence_idx 2 \
	&> ./results_cs_wildplaces_sgv_metloc_success_qcat.txt &

python evaluation/sgv_evaluation/eval_logg3d_sgv.py \
	--eval_pipeline LOGG3D \
	--eval_dataset GeneralDataset \
	--eval_dataset_name cs_wildplaces \
	--checkpoint_name checkpoints/logg3dnet_cs-wild-places.pth \
	--gp_rem False \
	--feature_dim 32 \
	--voxel_size 0.50 \
	--radius 10 30 \
	--d_thresh 0.4 \
	--local_max_eval_threshold 30 \
	--num_points 150000 \
	--icp_refine \
	--sequence_idx 3 \
	&> ./results_cs_wildplaces_sgv_metloc_success_samford.txt &

wait
echo Done all!