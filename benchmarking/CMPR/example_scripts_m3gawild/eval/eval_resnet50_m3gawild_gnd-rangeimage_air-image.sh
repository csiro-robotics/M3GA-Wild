#!/bin/bash

pixi run python evaluate_m3gawild/evaluate.py \
    --expid exp_m3gawild_range_res50 \
    --ckpt weights/resnet50_gnd-rangeimage_air-image.pth \
    --save_dir eval_outputs/resnet50_m3gawild_gnd-rangeimage_air-image \

