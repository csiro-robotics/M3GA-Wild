#!/bin/bash
source env-setup.sh

cd eval/
python evaluate_m3gawild.py \
    --config ../config/config_m3ga-wild.yaml \
    --model_config ../models/UMF/UMFnet_80k_img288.yml \
    --weights ../weights/umf_img288_m3ga-wild.pth
