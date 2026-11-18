#!/bin/bash
cd training/
pixi run python train.py \
    --config ../config/LPR/low_altitude/config_hotformerloc_m3ga-wild.txt \
    --model_config ../models/cfg_files/LPR/hotformerloc_cs-wild-places_cfg.txt
