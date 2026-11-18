#!/bin/bash

pixi run python evaluate_m3gawild/evaluate.py \
    --expid exp_m3gawild_lidar_hotformerloc_dinov2_frozen \
    --ckpt weights/dinov2_gnd-lidar-hotformerloc_air-image_frozen.pth \
    --save_dir eval_outputs/dinov2_m3gawild_gnd-lidar-hotformerloc_air-image_frozen \

