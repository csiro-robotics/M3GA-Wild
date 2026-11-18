#!/bin/bash
pixi run python evaluation/eval_inter_sequence_m3gawild.py \
    --config configs/m3gawild/BoQ_pos15m_onlyga.yaml \
    --pretrained_ckpt weights/BoQ_m3ga-wild_pos15m_onlyga_epoch39.ckpt \
    --save_dir eval_outputs/ 
