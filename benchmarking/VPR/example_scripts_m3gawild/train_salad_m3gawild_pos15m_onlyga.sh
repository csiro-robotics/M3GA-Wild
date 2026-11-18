#!/bin/bash
pixi run python train.py \
    --config configs/m3gawild/SALAD_pos15m_onlyga.yaml \
    --save_dir runs/SALAD \
    --pretrained_ckpt weights/SALAD_pretrained.pth
