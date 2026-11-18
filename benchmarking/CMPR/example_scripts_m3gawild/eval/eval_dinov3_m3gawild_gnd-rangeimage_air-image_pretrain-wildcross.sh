#!/bin/bash

pixi run python evaluate_m3gawild/evaluate.py \
    --expid exp_m3gawild_range_dinov3 \
    --ckpt weights/dinov3_gnd-rangeimage_air-image_pretrain-wildcross.pth \
    --save_dir eval_outputs/dinov3_m3gawild_gnd-rangeimage_air-image_pretrain-wildcross \

