# LiDAR and Multi-Modal Benchmarking and Evaluation
This subfolder contains scripts for training and evaluating LiDAR Place Recognition (VPR) and Multi-modal Place Recognition (MMPR) methods for **M3GA-Wild**.  We provide support for the following methods:

| Method | Modality | Paper | GitHub |
|:-|:-:|:-:|:-:|
| HOTFormerLoc | LPR | [Link](https://openaccess.thecvf.com/content/CVPR2025/papers/Griffiths_HOTFormerLoc_Hierarchical_Octree_Transformer_for_Versatile_Lidar_Place_Recognition_Across_CVPR_2025_paper.pdf) | [Source](https://github.com/csiro-robotics/HOTFormerLoc) |
| HOTFLoc++ | LPR | [Link](https://arxiv.org/abs/2511.09170) | [Source](https://github.com/csiro-robotics/HOTFLoc) |
| MinkLoc3Dv2 | LPR | [Link](https://arxiv.org/abs/2203.00972) | [Source](https://github.com/jac99/MinkLoc3Dv2) |
| EgoNN | LPR | [Link](https://arxiv.org/abs/2110.12486) | [Source](https://github.com/jac99/Egonn) |
| LoGG3D-Net | LPR | [Link](https://arxiv.org/abs/2109.08336) | [Source](https://github.com/csiro-robotics/LoGG3D-Net) |
| MinkLoc++ | MMPR | [Link](https://arxiv.org/abs/2104.05327) | [Source](https://github.com/jac99/MinkLocMultimodal) |
| UMF | MMPR | [Link](https://arxiv.org/abs/2403.13395) | [Source](https://github.com/DLR-RM/UMF) |

## Setup

### Dataset 
To download the **M3GA-Wild** Dataset, follow the [instructions](../../README.md#download-instructions) in the root directory of this repository. By default this repository will use the pre-shrunk images (*i.e.* images in the `images_shrunk` folder) to optimise the speed at which the raw data is loaded from the disk, but this can be changed by editing `IMAGE_DIR` in [this script](./dataset/M3GAWild/generate_train_test_tuples_6dof_images.py). The full-resolution point clouds can also be used by editing the `--cloud_dir` parameter of the [generation scripts](generate_m3ga-wild_low_altitude_tuples.sh).

### Environment

#### Pixi
With pixi, environment setup is as simple as:
```bash
pixi install
pixi run <script_you_want_to_run>
```
Pixi handles all the dependencies and environment variable setup required to run most of the LPR and MMPR experiments.

If you also need MinkowskiEngine dependencies (which you likely will), they can be installed with:
```bash
pixi run install-minkowskiengine
```
This command only needs to be run once to install MinkowskiEngine in your environment.

#### Mamba/Conda
```bash
# Note: replace 'mamba' with 'conda' if using a vanilla conda install
mamba create -n hotfloc++ python=3.11 -c conda-forge -y
mamba activate hotfloc++
mamba install 'numpy<2.0' -c conda-forge -y
mamba install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge -y
pip install -r requirements.txt
pip install --no-build-isolation libs/dwconv

# For metric loc:
pip install --no-build-isolation -e libs/geotransformer/

# NOTE: MinkowskiEngine can be optionally installed, but is only required for evaluating Minkowski-based models (EgoNN, MinkLoc3Dv2, MinkLoc++). See https://github.com/NVIDIA/MinkowskiEngine for installation instructions. For help installing on systems with CUDA>=12.0, please refer to these issues:
# https://github.com/NVIDIA/MinkowskiEngine/issues/543#issuecomment-1773458776
# https://github.com/NVIDIA/MinkowskiEngine/issues/621.
```

Modify the `PYTHONPATH` environment variable to include the absolute path to the repository root folder (ensure this variable is set every time you open a new shell): 
```bash
export PYTHONPATH=$PYTHONPATH:<path/to/M3GA-Wild/benchmarking/LPR+MMPR>
```
**NOTE:** Above step is handled automatically by running `env-setup.sh`, and is not needed at all for Pixi environment.

### Train/Eval Pickles
Before training or evaluating on M3GA-Wild, first generate the required training and evaluation tuples for low/high altitude LiDAR with:
```bash
pixi run ./generate_m3ga-wild_low_altitude_tuples.sh
pixi run ./generate_m3ga-wild_high_altitude_tuples.sh
```

This will place the required pickles into the M3GA-Wild data directory.

### Checkpoints
We provide download links for our fine-tuned checkpoints for each method, place these in [weights/](weights/) in the corresponding [LPR](weights/LPR/) or [MMPR](weights/MMPR/) folders:
| Method | Modality | Link |
|:-|:-|:-:|
| HOTFormerLoc | LPR (low altitude) | [Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/LPR/low_altitude/hotformerloc_m3ga-wild.pth?download=true) |
| HOTFormerLoc | LPR (high altitude) | [Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/blob/main/LPR/high_altitude/hotformerloc_m3ga-wild_airborne.pth?download=true) |
| HOTFLoc++ | LPR (low altitude) | [Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/LPR/low_altitude/hotfloc%2B%2B_m3ga-wild_voxel0.4m_stage1.pth?download=true) |
| HOTFLoc++ | LPR (high altitude) | [Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/LPR/high_altitude/hotfloc%2B%2B_m3ga-wild_airborne_stage2.pth?download=true) |
| MinkLoc3Dv2 | LPR (low altitude) | [Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/LPR/low_altitude/minkloc3dv2_m3ga-wild.pth?download=true) |
| MinkLoc3Dv2 | LPR (high altitude) | [Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/LPR/high_altitude/minkloc3dv2_m3ga-wild_airborne.pth?download=true) |
| EgoNN | LPR (low altitude) | [Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/LPR/low_altitude/egonn_m3ga-wild.pth?download=true) |
| EgoNN | LPR (high altitude) | [Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/LPR/high_altitude/egonn_m3ga-wild_airborne.pth?download=true) |
| MinkLoc++ (ResNet18) | MMPR | [Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/MMPR/minklocmultimodal_resnet18_m3ga-wild.pth?download=true) |
| MinkLoc++v2 (ResNet50) | MMPR | [Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/MMPR/minklocmultimodalv2_resnet50_m3ga-wild.pth?download=true) |
| MinkLoc++v2 (DINOV2) | MMPR | [Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/MMPR/minklocmultimodalv2_dinov2_m3ga-wild.pth?download=true) |
| MinkLoc++v2 (DINOV2 + modal dropout) | MMPR | [Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/MMPR/minklocmultimodalv2_dinov2_m3ga-wild_modal-dropout.pth?download=true) |

## Training and Evaluation
We provide example scripts for training and evaluating LPR and MMPR models in the `example_scripts_m3gawild/` directory. Ensure the M3GA-Wild dataset has been setup following the instructions in the [root README](../../README.md) first. For other models (LoGG3D-Net, UMF), we provide the altered third party code required to reproduce our experiments in `third_party_experiments/`. Refer to the respective READMEs in that directory for model-specific instructions. 

## Acknowledgements

Special thanks to the authors of [MinkLoc3Dv2](https://github.com/jac99/MinkLoc3Dv2), [OctFormer](https://github.com/octree-nn/octformer), [SpectralGV](https://github.com/csiro-robotics/SpectralGV), and [GeoTransformer](https://github.com/qinzheng93/GeoTransformer) for their excellent code, which formed the foundation of this codebase. Additional thanks to the authors of [LoGG3D-Net](https://github.com/csiro-robotics/LoGG3D-Net) and [UMF](https://github.com/DLR-RM/UMF).
