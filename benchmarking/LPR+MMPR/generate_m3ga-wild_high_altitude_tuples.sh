#!/usr/bin/env bash
set -euo pipefail

python dataset/M3GAWild/generate_train_test_tuples_6dof_images.py \
    --root data/M3GA-Wild \
    --cloud_dir clouds_voxel_0.80m_rmground \
    --aerial_type airborne \
    --eval_thresh 30 \
    --pos_thresh 15 \
    --neg_thresh 50 \
    --buffer_thresh 30 \
    --ignore_positives_poses \
    --splits Karawatha Venman QCAT Samford \
    && \
python dataset/M3GAWild/generate_train_test_tuples_6dof_images.py \
    --root data/M3GA-Wild \
    --cloud_dir clouds_voxel_0.80m_rmground \
    --aerial_type airborne \
    --eval_thresh 30 \
    --pos_thresh 15 \
    --neg_thresh 50 \
    --buffer_thresh 30 \
    --ignore_positives_poses \
    --ground_aerial_positives_only \
    --splits Karawatha Venman QCAT Samford
