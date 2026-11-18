#!/bin/bash
cd eval/
pixi run python evaluate_metric_loc_splits_rerank.py \
    --config ../config/LPR/high_altitude/config_minkloc3dv2_m3ga-wild-airborne.txt \
    --model_config ../models/cfg_files/LPR/minkloc3dv2.txt \
    --weights ../weights/LPR/high_altitude/minkloc3dv2_m3ga-wild_airborne.pth \
    --only_global
