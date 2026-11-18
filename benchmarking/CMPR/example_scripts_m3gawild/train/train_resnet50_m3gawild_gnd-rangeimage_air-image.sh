#!/bin/bash

pixi run python trainer_m3gawild.py \
    --expid exp_m3gawild_range_res50 \
    --save_dir train_outputs/resnet50_m3gawild_gnd-rangeimage_air-image/
