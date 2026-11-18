#!/bin/bash
cd training/
pixi run python train.py \
    --config ../config/LPR/low_altitude/config_hotfloc++_m3ga-wild_stage2.txt \
    --model_config ../models/cfg_files/LPR/hotfloc++_cs-wild-places_stage2_cfg.txt \
    --finetune_from ../weights/LPR/low_altitude/hotfloc++_m3ga-wild_voxel0.4m_stage1.pth
