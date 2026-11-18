#!/bin/bash
pixi run python evaluation/eval_inter_sequence_m3gawild.py \
    --config configs/m3gawild/AnyLoc_unstructured.yaml \
    --save_dir eval_outputs/
