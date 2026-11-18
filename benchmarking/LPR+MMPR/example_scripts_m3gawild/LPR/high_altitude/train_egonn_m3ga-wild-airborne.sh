#!/bin/bash
cd training/
pixi run python train.py \
    --config ../config/LPR/high_altitude/config_egonn_m3ga-wild-airborne.txt \
    --model_config ../models/cfg_files/LPR/egonn.txt
