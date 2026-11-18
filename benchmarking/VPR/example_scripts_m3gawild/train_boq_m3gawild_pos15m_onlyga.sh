#!/bin/bash
pixi run python train.py \
    --config configs/m3gawild/BoQ_pos15m_onlyga.yaml \
    --save_dir runs/BoQ \
    --pretrained_ckpt weights/BoQ_pretrained.pth
