#!/bin/bash
cd training/
pixi run python train.py \
    --config ../config/MMPR/config_minklocmultimodal_m3ga-wild_img224.txt \
    --model_config ../models/cfg_files/MMPR/minklocmultimodal.txt
