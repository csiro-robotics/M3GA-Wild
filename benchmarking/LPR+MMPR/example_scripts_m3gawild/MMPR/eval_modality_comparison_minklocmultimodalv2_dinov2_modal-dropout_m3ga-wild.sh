#!/bin/bash
cd eval/
pixi run python pnv_evaluate_splits_multimodal.py \
    --config ../config/MMPR/config_minklocmultimodal_m3ga-wild_img336_modal-dropout.txt \
    --model_config ../models/cfg_files/MMPR/minklocmultimodalv2_dinov2.txt \
    --weights ../weights/MMPR/minklocmultimodalv2_dinov2_m3ga-wild_modal-dropout.pth \
    --gps_radius 0.0
