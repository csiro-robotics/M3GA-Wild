#!/bin/bash

pixi run python trainer_m3gawild.py \
    --expid exp_m3gawild_heightmap_chmv2_dinov2_heightmap336 \
    --save_dir train_outputs/dinov2_m3gawild_gnd-heightmap336_air-chmv2/
