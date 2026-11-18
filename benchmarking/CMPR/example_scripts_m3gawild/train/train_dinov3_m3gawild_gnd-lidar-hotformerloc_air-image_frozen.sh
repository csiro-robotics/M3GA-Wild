#!/bin/bash

pixi run python trainer_m3gawild.py \
    --expid exp_m3gawild_lidar_hotformerloc_dinov3_frozen \
    --save_dir train_outputs/dinov3_m3gawild_gnd-lidar-hotformerloc_air-image_frozen
