#!/bin/bash
cd training/
pixi run python train.py \
    --config ../config/LPR/low_altitude/config_egonn_m3ga-wild.txt \
    --model_config ../models/cfg_files/LPR/egonn.txt
