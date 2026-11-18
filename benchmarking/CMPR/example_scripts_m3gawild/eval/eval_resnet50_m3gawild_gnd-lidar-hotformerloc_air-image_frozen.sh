#!/bin/bash

pixi run python evaluate_m3gawild/evaluate.py \
    --expid exp_m3gawild_lidar_hotformerloc_res50_frozen \
    --ckpt weights/resnet50_gnd-lidar-hotformerloc_air-image_frozen.pth \
    --save_dir eval_outputs/resnet50_m3gawild_gnd-lidar-hotformerloc_air-image_frozen \

