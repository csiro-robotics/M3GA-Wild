#!/bin/bash

pixi run python trainer_m3gawild.py \
    --expid exp_m3gawild_lidar_hotformerloc_dinov2_frozen \
    --save_dir train_outputs/dinov2_m3gawild_gnd-lidar-hotformerloc_air-image_frozen
