#!/bin/bash

pixi run python evaluate_m3gawild/evaluate.py \
    --expid exp_m3gawild_heightmap_chmv2_res50_heightmap224 \
    --ckpt weights/resnet50_gnd-heightmap224_air-chmv2.pth \
    --save_dir eval_outputs/resnet50_m3gawild_gnd-heightmap224_air-chmv2/ \
