#!/bin/bash
cd training/
pixi run python train.py \
    --config ../config/LPR/low_altitude/config_minkloc3dv2_m3ga-wild.txt \
    --model_config ../models/cfg_files/LPR/minkloc3dv2.txt
