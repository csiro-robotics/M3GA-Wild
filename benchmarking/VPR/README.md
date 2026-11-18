# Visual Place Recognition: Benchmarking and Evaluation

This subfolder contains scripts for training and evaluating Visual Place Recognition (VPR) methods for **M3GA-Wild**.  We provide support for the following VPR methods:

| Method | Paper | GitHub |
|:-|:-:|:-:|
| NetVLAD | [Link](https://openaccess.thecvf.com/content_cvpr_2016/papers/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.pdf) | [Source](https://github.com/gmberton/VPR-methods-evaluation) |
| MixVPR | [Link](https://openaccess.thecvf.com/content/WACV2023/papers/Ali-bey_MixVPR_Feature_Mixing_for_Visual_Place_Recognition_WACV_2023_paper.pdf) | [Source](https://github.com/amaralibey/MixVPR) |
| SALAD | [Link](https://openaccess.thecvf.com/content/CVPR2024/papers/Izquierdo_Optimal_Transport_Aggregation_for_Visual_Place_Recognition_CVPR_2024_paper.pdf) | [Source](https://github.com/serizba/salad) |
| BoQ | [Link](https://openaccess.thecvf.com/content/CVPR2024/papers/Ali-bey_BoQ_A_Place_is_Worth_a_Bag_of_Learnable_Queries_CVPR_2024_paper.pdf) | [Source](https://github.com/amaralibey/Bag-of-Queries) |
| AnyLoc | [Link](https://arxiv.org/pdf/2308.00688) | [Source](https://github.com/AnyLoc/AnyLoc) |

## Setup 

### Dataset 
To download the **M3GA-Wild** Dataset, follow the [instructions](../../README.md#download-instructions) in the root directory of this repository.  By default this repository will use the pre-shrunk images (*i.e.* images in the `images_shrunk` folder) to optimise the speed at which the raw data is loaded from the disk, but this can be changed by editing `./data/dataset.py`.

Once the dataset is downloaded, please edit the `m3gawild_path` parameter in the config file used for training (found in the [configs/m3gawild](configs/m3gawild) folder) to point towards the **M3GA-Wild** dataset (**NOTE**: an absolute path is required currently). 

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

### Checkpoints
We provide download links for both the author-provided pre-trained checkpoints on urban environments and our fine-tuned checkpoints for each method, place these in [weights/](weights/):

| Method | Type | Link |
|:-|:-|:-:|
|NetVLAD|Pre-trained|Downloaded by training script|
||Fine-tuned|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/VPR/NetVLAD_pos15m_onlyga_epoch39.ckpt?download=true)|
|MixVPR|Pre-trained|[Download](https://huggingface.co/CSIRORobotics/WildCross/resolve/main/VPR/MixVPR/pretrained.pth)|
||Fine-tuned|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/VPR/MixVPR_pos15m_onlyga_epoch39.ckpt?download=true)|
|SALAD|Pre-trained|[Download](https://huggingface.co/CSIRORobotics/WildCross/resolve/main/VPR/SALAD/pretrained.pth)|
||Fine-tuned|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/VPR/SALAD_pos15m_onlyga_epoch39.ckpt?download=true)|
|BoQ|Pre-trained|[Download](https://huggingface.co/CSIRORobotics/WildCross/resolve/main/VPR/BoQ/pretrained.pth)|
||Fine-tuned|[Download](https://huggingface.co/EGriffiths-CSIRO/M3GA-Wild/resolve/main/VPR/BoQ_pos15m_onlyga_epoch39.ckpt?download=true)|
|AnyLoc|Pre-trained|Downloaded by training script|

## Training
To train a given method on the M3GA-Wild Dataset, run `train.py` as follows:
```bash
pixi run python train.py \
    --config /path/to/method/configs.yaml \
    --save_dir /path/to/save/dir \
    --pretrained_ckpt /path/to/checkpoint.pth
```

Where:
- `--config` is for providing the config file for a given method, which can be found in `./configs/m3gawild`
- `--savedir` is for providing the directory to save checkpoints and logs in
- `--pretrained_ckpt` is for providing a path to a pre-trained checkpoint to load before starting training.  For best performance, we recommend using the urban pre-trained checkpoints provided here.

It is also possible to modify arguments in the config file through the command line by passing the name of the argument and its value as a pair of inputs at the end of the command, (*e.g.* `max_epochs 20` to modify the maximum number of training epochs) 

We additionally provide example training scripts for all the models in `example_scripts_m3gawild/`.

## Evaluation
For evaluating a VPR model, run `evaluation/eval_inter_sequence_m3gawild.py`.
```bash
pixi run python evaluation/eval_inter_sequence_m3gawild.py \
    --config /path/to/method/configs.yaml \
    --save_dir /path/to/save/dir \
    --pretrained_ckpt /path/to/checkpoint.pth
```

We additionally provide example evaluation scripts for all the models in `example_scripts_m3gawild/`.


## Acknowledgements
We would like to thank the authors of [WildCross](https://github.com/csiro-robotics/WildCross/) for their dataset and evaluation code, which provided a foundation for the M3GA-Wild dataset and VPR/CMPR evaluations. We would also like to thank the authors and open source maintainers for NetVLAD, [MixVPR](https://github.com/amaralibey/MixVPR), [SALAD](https://github.com/serizba/salad), [BOQ](https://github.com/amaralibey/Bag-of-Queries), [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation), and [AnyLoc](https://github.com/AnyLoc/AnyLoc), which were invaluable for the benchmarking experiments and code used in this paper and repository.