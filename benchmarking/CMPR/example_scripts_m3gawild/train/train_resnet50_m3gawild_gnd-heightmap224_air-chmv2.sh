#!/bin/bash

pixi run python trainer_m3gawild.py \
    --expid exp_m3gawild_heightmap_chmv2_res50_heightmap224 \
    --save_dir train_outputs/resnet50_m3gawild_gnd-heightmap224_air-chmv2/
