from typing import Union, Optional
import glob
import albumentations as A
import numpy as np
import torch
from torchvision.transforms import v2
from .augmentation import TrainTransform, ValTransform


def get_filenames(sequences, data_path, length=None):
    filenames = np.array([])
    for sequence in sorted(sequences):
        # populate timestamp files in sequence in format <Sequence digit>/<timestampfilename>
        folder = f"{data_path}/{sequence}/velodyne/"
        files = glob.glob(folder+'*bin')
        image_arr = np.arange(0, len(files)).astype(str)
        image_ids = np.char.zfill(image_arr,6)
        image_ids = np.core.defchararray.add(f"{sequence}/", image_ids)
        filenames = np.append(filenames,image_ids)

        if(length is not None):
            filenames = filenames[0:length]
    return filenames

def get_transforms(mode, size):
    if mode == "train":
        return A.Compose(
            [   
                A.Resize(size, size),
                A.Normalize(max_pixel_value=255.0),
                A.CoarseDropout(always_apply=False, p=0.9, max_holes=5, max_height=100, max_width=100, min_height=50, min_width=50, fill_value=(0, 0, 0), mask_fill_value=None),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(size, size),
                A.Normalize(max_pixel_value=255.0),
            ]
        )

def get_transforms_m3gawild(
    mode,
    image_size: Union[int, tuple[int, int]],
    image_centre_crop: bool,
    range_image_size: Union[int, tuple[int, int]],
    aug_mode: int = 1,
    lidar_aug_mode: int = 2,
    sat493m_norm: bool = False,
) -> dict:
    """
    Using torchvision transforms to allow rectangular resize ratios.
    """
    if aug_mode not in (1, 2):
        raise ValueError()
    if sat493m_norm:
        img_mean = (0.420, 0.411, 0.296)
        img_std = (0.213, 0.156, 0.143)
    else:
        img_mean = (0.485, 0.456, 0.406)
        img_std = (0.229, 0.224, 0.225)
    if mode == "train":
        if aug_mode == 1:
            image_transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(image_size, antialias=True),
                    (v2.CenterCrop(image_size) if image_centre_crop else v2.Identity()),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=img_mean,
                        std=img_std,
                    ),
                    v2.RandomErasing(p=0.9),
                    v2.RandomApply([v2.RandomErasing(p=1.0)], p=0.3),
                ]
            )
            range_transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(range_image_size, antialias=True),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=img_mean,
                        std=img_std,
                    ),
                    v2.RandomErasing(p=0.9),
                    v2.RandomApply([v2.RandomErasing(p=1.0)], p=0.3),
                ]
            )
        elif aug_mode == 2:  # More severe augmentations
            image_transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(image_size, antialias=True),
                    (v2.CenterCrop(image_size) if image_centre_crop else v2.Identity()),
                    v2.RandAugment(num_ops=2, magnitude=9, interpolation=2),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=img_mean,
                        std=img_std,
                    ),
                    v2.RandomErasing(p=0.5, scale=(0.02, 0.15)),
                ]
            )
            range_transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(range_image_size, antialias=True),
                    v2.RandomAffine(degrees=10., translate=(0.0, 0.10), scale=(0.9, 1.0), shear=5., interpolation=2),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=img_mean,
                        std=img_std,
                    ),
                    v2.RandomErasing(p=0.5, scale=(0.02, 0.15)),
                ]
            )
        lidar_transform = TrainTransform(
            aug_mode=lidar_aug_mode,
            normalize_points=True,
            scale_factor=None,
            unit_sphere_norm=False,
            zero_mean=True,
            random_rot_theta=180,
        )
    else:
        image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(image_size, antialias=True),
                (v2.CenterCrop(image_size) if image_centre_crop else v2.Identity()),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=img_mean,
                    std=img_std,
                ),
            ]
        )
        range_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(range_image_size, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=img_mean,
                    std=img_std,
                ),
            ]
        )
        lidar_transform = ValTransform(
            normalize_points=True, scale_factor=None, unit_sphere_norm=False, zero_mean=True,
        )
    return {'image': image_transform, 'range_image': range_transform, 'lidar': lidar_transform}