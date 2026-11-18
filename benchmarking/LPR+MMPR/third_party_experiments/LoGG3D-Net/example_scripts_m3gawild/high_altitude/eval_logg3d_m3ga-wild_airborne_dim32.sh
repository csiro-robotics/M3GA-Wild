#!/bin/bash
pixi run python evaluation/evaluate_wildplaces.py \
    --eval_pipeline 'LOGG3D' \
	--eval_dataset 'GeneralDataset' \
	--eval_dataset_name "m3gawild_airborne" \
	--eval_batch_size "24" \
	--gp_rem False \
	--cs_wildplaces_dir ../../../../data/M3GA-Wild \
	--checkpoint_name "checkpoints/logg3dnet_m3ga-wild_airborne.pth" \
	--voxel_size 0.50 \
	--feature_dim 32
