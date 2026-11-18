#!/bin/bash

pixi run python trainer_m3gawild.py \
    --expid exp_m3gawild_lidar_hotformerloc_res50_frozen \
    --save_dir train_outputs/resnet50_m3gawild_gnd-lidar-hotformerloc_air-image_frozen
