#!/bin/bash
cd training/
pixi run python train.py \
    --config ../config/MMPR/config_minklocmultimodal_m3ga-wild_img288.txt \
    --model_config ../models/cfg_files/MMPR/minklocmultimodalv2_resnet50.txt
