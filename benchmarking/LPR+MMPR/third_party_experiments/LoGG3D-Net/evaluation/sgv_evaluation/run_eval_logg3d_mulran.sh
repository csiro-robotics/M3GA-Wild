#!/bin/bash

python evaluation/sgv_evaluation/eval_logg3d_sgv.py \
	--eval_pipeline LOGG3D \
	--eval_dataset GeneralDataset \
	--eval_dataset_name mulran \
	--checkpoint_name checkpoints/logg3dnet_mulran.pth \
	--gp_rem False \
	--feature_dim 32 \
	--voxel_size 0.50 \
	--radius 5 20 \
	--d_thresh 0.4 \
	--local_max_eval_threshold 20 \
	--num_points 35000 \
	&> ./results_mulran_sgv_metloc_success.txt
