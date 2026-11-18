#!/bin/bash
cd eval/
pixi run python evaluate_metric_loc_splits_rerank.py \
    --config ../config/MMPR/config_minklocmultimodal_m3ga-wild_img224.txt \
    --model_config ../models/cfg_files/MMPR/minklocmultimodal.txt \
    --weights ../weights/MMPR/minklocmultimodal_resnet18_m3ga-wild.pth \
    --only_global
