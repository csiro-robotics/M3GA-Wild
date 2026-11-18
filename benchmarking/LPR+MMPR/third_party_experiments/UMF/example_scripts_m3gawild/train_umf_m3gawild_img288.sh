#!/bin/bash
source env-setup.sh

cd training/
python train.py \
    --config ../config/config_m3ga-wild.yaml \
    --model_config ../models/UMF/UMFnet_80k_img288.yml \
