# Cross-Modal Benchmarking and Evaluation

This subfolder contains scripts for training and evaluating Cross-Modal Place Recognition (CMPR) methods using the **M3GA-Wild** dataset. We provide support for the baseline cross-modal approach LIP-Loc, using the ResNet50, DINOv2 and DINOv3 backbones, and provide multiple combinations of cross-modal configurations for evaluation, e.g. ground range image -> aerial RGB, ground heightmap -> aerial RGB, ground LiDAR -> aerial RGB, etc.  

## Setup

### Dataset 
To download the **M3GA-Wild** Dataset, follow the [instructions](../../README.md#download-instructions) in the root directory of this repository. You will then need to update the `data_path` parameter of any config file in [config/](config/) to point to the M3GA-Wild dataset.  By default this repository will use the full resolution point clouds (`clouds`) and the pre-shrunk images and heightmaps (*i.e.* images in the `images_shrunk` folder, heightmaps in `chmv2_heightmaps_shrunk`) to optimise the speed at which the raw data is loaded from the disk, but this can be changed by editing the `image_folder` parameter of any config files you use. 

### Train/Eval Pickles
Before training or evaluating on M3GA-Wild, you will first generate the 'low altitude' training and evaluation tuples following the instructions in the [LPR+MMPR README](../LPR+MMPR/README.md#m3ga-wild-experiments), as the CMPR experiments use the same training/evaluation set.

### Environment
#### Pixi
We provide a [pixi](https://pixi.prefix.dev/latest/installation/) environment to simplify environment setup.

Simply run this command out of this directory:
```bash
pixi install
pixi run <script_you_want_to_run>
```
Pixi handles all the dependencies and environment variable setup required to run VPR training and evaluation.

#### Mamba/Conda
We alternatively provide an environment file for use with **mamba/conda**.  The environment can be installed by running the following command out of this directory:

```bash
mamba env create -f environment.yaml
```
You will then need to manually `source env-setup.sh` to correctly configure your PYTHONPATH in any new shell instance.

### DINOv3
To train with DINOv3, at time of publication, you need to clone the official repo and checkpoint path yourself.
See https://github.com/facebookresearch/dinov3 for details on the repo and the checkpoints. Our model uses the vits16 pre-trained dinov3 model.
Once these are downloaded, you need to define `REPO_DIR` and `CHECKPOINT_PATH` within lines 106 and 107 of `benchmarking/cross_modal/models/CLIPModelV1NormM3GAWild.py`. Current default lines are shown below for convenience. You may also optionally define the path in lines 119 and 120 if you wish to use the DINOv3 model pre-trained on the SAT-493M dataset.

```bash
REPO_DIR="/path/to/dinov3"
CHECKPOINT_PATH="/path/to/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
```

### Checkpoints 
We provide download links for our fine-tuned checkpoints for each network backbone, place these in [weights/](weights/):

| Backbone | Ground Modality | Aerial Modality | Link |
|:-|:-|:-|:-:|
|ResNet50|Range Image|RGB|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/CMPR/resnet50_gnd-rangeimage_air-image.pth?download=true)|
|DinoV2|Range Image|RGB|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/CMPR/dinov2_gnd-rangeimage_air-image_pretrain-wildcross.pth?download=true)|
|DinoV3|Range Image|RGB|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/CMPR/dinov3_gnd-rangeimage_air-image_pretrain-wildcross.pth?download=true)|
|ResNet50|Height Map|RGB|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/CMPR/resnet50_gnd-heightmap224_air-image.pth?download=true)|
|DinoV2|Height Map|RGB|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/CMPR/dinov2_gnd-heightmap224_air-image.pth?download=true)|
|DinoV3|Height Map|RGB|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/CMPR/dinov3_gnd-heightmap336_air-image.pth?download=true)|
|ResNet50|Height Map|CHMv2 Height Map|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/CMPR/resnet50_gnd-heightmap224_air-chmv2.pth?download=true)|
|DinoV2|Height Map|CHMv2 Height Map|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/CMPR/dinov2_gnd-heightmap336_air-chmv2.pth?download=true)|
|DinoV3|Height Map|CHMv2 Height Map|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/CMPR/dinov3_gnd-heightmap336_air-chmv2.pth?download=true)|
|ResNet50 (+ HOTFormerLoc) |LiDAR|RGB|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/CMPR/resnet50_gnd-lidar-hotformerloc_air-image_frozen.pth?download=true)|
|DinoV2 (+ HOTFormerLoc) |LiDAR|RGB|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/CMPR/dinov2_gnd-lidar-hotformerloc_air-image_frozen.pth?download=true)|
|DinoV3 (+ HOTFormerLoc) |LiDAR|RGB|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/CMPR/dinov3_gnd-lidar-hotformerloc_air-image_frozen.pth?download=true)|

> **NOTE**: If using any of the HOTFormerLoc variants, you will need to download the original weights pre-trained on Wild-Places from [this link](https://www.dropbox.com/scl/fi/yd94iy9dq6k1m312ifnyx/hotformerloc_wild-places.pth?rlkey=5ndv0p48c7hyjvah90eab1l1e&st=zl1716hh&dl=0). You may also need weights from [WildCross](https://github.com/csiro-robotics/WildCross/) for some of the pre-trained models (e.g. 'dinov2_split_1.pth', 'dinov3_split_1.pth').

## Training
To train a given backbone on the M3GA-Wild Dataset, run `trainer_m3gawild.py` as follows:

```bash
pixi run python trainer_m3gawild.py \
    --expid $EXPID \
    --save_dir /path/to/save_dir \
```

Where:
- `--expid` is the name of the config file for this experiment.  Available config files can be found in `config`.
- `--save_dir` is for the directory which the training logs and checkpoints will be saved to

Final model will be saved in `save_dir` under `latest.pth`. Ensure the `data_path` to M3GA-Wild is updated in the config file you use.

We additionally provide example training scripts in `example_scripts_m3gawild/train/`, for training each configuration
reported in the paper.

## Evaluation
To evaluate a given checkpoint, run `evaluate_m3gawild/evaluate.py` as follows:

```bash
pixi run python evaluate_m3gawild/evaluate.py \
    --expid $EXPID \
    --save_dir /path/to/save_dir \
    --ckpt /path/to/checkpoint.pth \
```

Where:
- `--expid` and `--save_dir` are the same as for the training script
- `--ckpt` is for setting the path to the checkpoint to be evaluated 

We additionally provide example training scripts in `example_scripts_m3gawild/eval/`, for evaluating each configuration
reported in the paper.

## Acknowledgements
We would like to thank the authors of [WildCross](https://github.com/csiro-robotics/WildCross/) for their dataset and evaluation code, which provided a foundation for the M3GA-Wild dataset and VPR/CMPR evaluations. We would also like to thank the authors of the original LIP-Loc paper and the maintainers of the open source repository hosted at [https://github.com/Shubodh/lidar-image-pretrain-VPR](https://github.com/Shubodh/lidar-image-pretrain-VPR) which is used as the basis for the code in this repository.