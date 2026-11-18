#!/bin/bash
pixi run python train.py \
    --config configs/m3gawild/MixVPR_pos15m_onlyga.yaml \
    --save_dir runs/MixVPR \
    --pretrained_ckpt weights/MixVPR_pretrained.pth
