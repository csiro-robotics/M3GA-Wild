#!/bin/bash
cd training/
pixi run python train.py \
    --config ../config/LPR/high_altitude/config_hotformerloc_m3ga-wild-airborne.txt \
    --model_config ../models/cfg_files/LPR/hotformerloc_cs-wild-places_cfg.txt
