#!/bin/bash

pixi run python trainer_m3gawild.py \
    --expid exp_m3gawild_range_dinov2 \
    --save_dir train_outputs/dinov2_m3gawild_gnd-rangeimage_air-image_pretrain-wildcross/ \
    --pretrained weights/dinov2_split_1.pth
