#!/bin/bash
cd eval/
pixi run python evaluate_metric_loc_splits_rerank.py \
    --config ../config/LPR/low_altitude/config_egonn_m3ga-wild.txt \
    --model_config ../models/cfg_files/LPR/egonn.txt \
    --weights ../weights/LPR/low_altitude/egonn_m3ga-wild.pth \
    --only_global
