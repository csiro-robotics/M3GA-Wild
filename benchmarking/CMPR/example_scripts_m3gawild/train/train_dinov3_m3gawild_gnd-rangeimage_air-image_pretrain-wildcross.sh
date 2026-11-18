#!/bin/bash

pixi run python trainer_m3gawild.py \
    --expid exp_m3gawild_range_dinov3 \
    --save_dir train_outputs/dinov3_m3gawild_gnd-rangeimage_air-image_pretrain-wildcross/ \
    --pretrained weights/dinov3_split_1.pth
