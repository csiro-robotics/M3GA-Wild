#!/bin/bash

pixi run python evaluate_m3gawild/evaluate.py \
    --expid exp_m3gawild_heightmap_dinov3_heightmap336 \
    --ckpt weights/dinov3_gnd-heightmap336_air-image.pth \
    --save_dir eval_outputs/dinov3_m3gawild_gnd-heightmap336_air-image \

