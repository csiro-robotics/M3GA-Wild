#!/bin/bash

pixi run python evaluate_m3gawild/evaluate.py \
    --expid exp_m3gawild_heightmap_dinov2_heightmap224 \
    --ckpt weights/dinov2_gnd-heightmap224_air-image.pth \
    --save_dir eval_outputs/dinov2_m3gawild_gnd-heightmap224_air-image/ \
