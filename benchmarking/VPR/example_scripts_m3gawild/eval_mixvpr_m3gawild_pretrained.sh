#!/bin/bash
pixi run python evaluation/eval_inter_sequence_m3gawild.py \
    --config configs/m3gawild/MixVPR.yaml \
    --pretrained_ckpt weights/MixVPR_pretrained.pth \
    --save_dir eval_outputs/ 
