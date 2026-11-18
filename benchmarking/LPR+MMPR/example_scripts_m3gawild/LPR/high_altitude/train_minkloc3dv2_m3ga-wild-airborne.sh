#!/bin/bash
cd training/
pixi run python train.py \
    --config ../config/LPR/high_altitude/config_minkloc3dv2_m3ga-wild-airborne.txt \
    --model_config ../models/cfg_files/LPR/minkloc3dv2.txt
