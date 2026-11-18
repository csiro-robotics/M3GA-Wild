#!/bin/bash
cd eval/
pixi run python evaluate_metric_loc_splits_rerank.py \
    --config ../config/LPR/low_altitude/config_hotformerloc_m3ga-wild.txt \
    --model_config ../models/cfg_files/LPR/hotformerloc_cs-wild-places_cfg.txt \
    --weights ../weights/LPR/low_altitude/hotformerloc_m3ga-wild.pth \
    --only_global
