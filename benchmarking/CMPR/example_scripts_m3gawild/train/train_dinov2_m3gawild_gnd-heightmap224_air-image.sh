#!/bin/bash

pixi run python trainer_m3gawild.py \
    --expid exp_m3gawild_heightmap_dinov2_heightmap224 \
    --save_dir train_outputs/dinov2_m3gawild_gnd-heightmap224_air-image/
