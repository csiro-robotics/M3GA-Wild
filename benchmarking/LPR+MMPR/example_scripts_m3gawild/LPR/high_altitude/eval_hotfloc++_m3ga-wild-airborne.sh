#!/bin/bash
cd eval/
pixi run python evaluate_metric_loc_splits_rerank.py \
    --config ../config/LPR/high_altitude/config_hotfloc++_m3ga-wild-airborne_stage2.txt \
    --model_config ../models/cfg_files/LPR/hotfloc++_cs-wild-places_stage2_cfg.txt \
    --weights ../weights/LPR/high_altitude/hotfloc++_m3ga-wild_airborne_stage2.pth \
    --only_global
