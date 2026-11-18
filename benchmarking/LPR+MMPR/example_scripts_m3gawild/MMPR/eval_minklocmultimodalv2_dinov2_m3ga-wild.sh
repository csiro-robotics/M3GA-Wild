#!/bin/bash
cd eval/
pixi run python evaluate_metric_loc_splits_rerank.py \
    --config ../config/MMPR/config_minklocmultimodal_m3ga-wild_img336.txt \
    --model_config ../models/cfg_files/MMPR/minklocmultimodalv2_dinov2.txt \
    --weights ../weights/MMPR/minklocmultimodalv2_dinov2_m3ga-wild.pth \
    --only_global
