from typing import Optional
import numpy as np
import math
from scipy.linalg import expm, norm
import random
import torch
import torchvision.transforms as transforms


class TrainTransform:
    def __init__(self, aug_mode):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        t = []
        if self.aug_mode == 0:
            pass
        elif self.aug_mode == 1:
            t.extend([
                JitterPoints(sigma=0.001, clip=0.002),
                RemoveRandomPoints(r=(0.15, 0.3)),
                RandomRotation(axis=np.array([0, 0, 1]), max_theta=180), 
                RandomTranslation(max_delta=0.02), 
                RemoveRandomBlock(p=0.5),
            ])
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class TrainRGBTransform:
    def __init__(self, aug_mode, image_size=(224, 224)):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        if self.aug_mode == 0:
            t = [transforms.Resize(image_size, antialias=True),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        elif self.aug_mode == 1:
            t = [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 #transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=0.1),
                 #transforms.RandomHorizontalFlip(p=0.5),
                 transforms.Resize(image_size, antialias=True),
                 transforms.ToTensor(),
                 transforms.RandomErasing(scale=(0.1, 0.4)),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)


    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class TrainGreyTransform:
    def __init__(self, aug_mode, image_size=(224, 224)):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        if self.aug_mode == 0:
            t = [transforms.Resize(image_size, antialias=True),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])]
        elif self.aug_mode == 1:
            t = [
                transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Resize(image_size, antialias=True),
                transforms.ToTensor(),
                transforms.RandomErasing(scale=(0.15, 0.4)),
                transforms.Normalize([0.5], [0.5])]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)


    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e

class ValRGBTransform:
    def __init__(self, image_size=(224, 224)):
        # 1 is default mode, no transform
        t = [transforms.Resize(image_size, antialias=True),
             transforms.ToTensor(), 
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        e = self.transform(e)
        return e

class ValGreyTransform:
    def __init__(self, image_size=(224, 224)):
        t = [transforms.Resize(image_size, antialias=True),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        e = self.transform(e)
        return e

class RandomFlip:
    def __init__(self, p):
        # p = [p_x, p_y, p_z] probability of flipping each axis
        assert len(p) == 3
        assert 0 < sum(p) <= 1, 'sum(p) must be in (0, 1] range, is: {}'.format(sum(p))
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, coords):
        r = random.random()
        if r <= self.p_cum_sum[0]:
            # Flip the first axis
            coords[..., 0] = -coords[..., 0]
        elif r <= self.p_cum_sum[1]:
            # Flip the second axis
            coords[..., 1] = -coords[..., 1]
        elif r <= self.p_cum_sum[2]:
            # Flip the third axis
            coords[..., 2] = -coords[..., 2]

        return coords


class RandomRotation:
    def __init__(self, axis=None, max_theta=180, max_theta2=15):
        self.axis = axis
        self.max_theta = max_theta      # Rotation around axis
        self.max_theta2 = max_theta2    # Smaller rotation in random direction

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(axis, (np.pi * self.max_theta / 180) * 2 * (np.random.rand(1) - 0.5))
        if self.max_theta2 is None:
            coords = coords @ R
        else:
            R_n = self._M(np.random.rand(3) - 0.5, (np.pi * self.max_theta2 / 180) * 2 * (np.random.rand(1) - 0.5))
            coords = coords @ R @ R_n

        return coords


class RandomTranslation:
    def __init__(self, max_delta=0.05):
        self.max_delta = max_delta

    def __call__(self, coords):
        trans = self.max_delta * np.random.randn(1, 3)
        return coords + trans.astype(np.float32)


class RandomScale:
    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s.astype(np.float32)


class RandomShear:
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, coords):
        T = np.eye(3) + self.delta * np.random.randn(3, 3)
        return coords @ T.astype(np.float32)


class JitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1.):
        assert 0 < p <= 1.
        assert sigma > 0.

        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, e):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """

        sample_shape = (e.shape[0],)
        if self.p < 1.:
            # Create a mask for points to jitter
            m = torch.distributions.categorical.Categorical(probs=torch.tensor([1 - self.p, self.p]))
            mask = m.sample(sample_shape=sample_shape)
        else:
            mask = torch.ones(sample_shape, dtype=torch.int64 )

        mask = mask == 1
        jitter = self.sigma * torch.randn_like(e[mask])

        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)

        e[mask] = e[mask] + jitter
        return e


class RemoveRandomPoints:
    def __init__(self, r):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e):
        n = len(e)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(range(n), size=int(n*r), replace=False)   # select elements to remove
        e[mask] = torch.zeros_like(e[mask])
        return e


class RandomScaling:
    def __init__(self, scale_range=(0.9, 1.1)):
        """
        Initializes the RandomScaling transformer with a specified scale range.
        
        Parameters:
        - scale_range: A tuple of two floats, specifying the minimum and maximum
                       scaling factors.
        """
        self.scale_range = scale_range

    def __call__(self, points):
        """
        Applies random scaling to the point cloud.
        
        Parameters:
        - points: A numpy array of shape (N, 3), where N is the number of points.
        
        Returns:
        - Scaled points as a numpy array of shape (N, 3).
        """
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        
        # Scale the points
        scaled_points = points * scale_factor
        
        return scaled_points

class RemoveRandomBlock:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)     # Fronto-parallel cuboid to remove
            mask = (x < coords[..., 0]) & (coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
            coords[mask] = torch.zeros_like(coords[mask])
        return coords


class TrainSetTransform:
    def __init__(self, aug_mode):
        self.aug_mode = aug_mode
        self.transform = None
        if aug_mode == 0:
            t = None
        elif aug_mode == 1:
            t = [RandomRotation(max_theta=5, max_theta2=0, axis=np.array([0, 0, 1])),
                 RandomFlip([0.25, 0.25, 0.])]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(aug_mode))
        if t is None:
            self.transform = None
        else:
            self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


def tensor2img(x):
    t = transforms.Compose([transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                            transforms.ToPILImage()])
    return t(x)


class Normalize:
    """
    From HOTFormerLoc.
    Normalize cloud within `[-norm_range, norm_range]`. Defaults to `[-1, 1]`.

    Alternatively, provide `scale_factor` to normalize the cloud with a fixed
    scaling factor. E.g. `cloud_normalized = (cloud - centroid) / scale_factor`.
    Also supports normlization within a unit sphere, and normalizing with and
    without shifting to zero mean.
    """    
    def __init__(self, norm_range: Optional[float] = None,
                 scale_factor: Optional[float] = None,
                 unit_sphere_norm: bool = False,
                 zero_mean: bool = True,
                 return_shift_and_scale: bool = False):
        assert not all([arg is not None for arg in [norm_range, scale_factor]]),\
            "Must specify one of norm_range or scale_factor, not both"
        self.norm_range = 0.999  # slightly under 1.0 to prevent floating point errors
        self.scale_factor = None
        self.unit_sphere_norm = unit_sphere_norm
        self.zero_mean = zero_mean
        self.return_shift_and_scale = return_shift_and_scale
        if norm_range is not None:
            assert norm_range > 0, "Range must be positive"
            self.norm_range = norm_range
        elif scale_factor is not None:
            assert scale_factor > 0, "Scale factor must be positive"
            self.norm_range = None
            self.scale_factor = scale_factor

    def __call__(self, coords: torch.Tensor):
        current_shift_and_scale = torch.zeros(4, dtype=coords.dtype)  # (shift_x, shift_y, shift_z, scale)
        if not self.unit_sphere_norm:                
            bbmin = coords.min(dim=0).values
            bbmax = coords.max(dim=0).values
            if self.zero_mean:
                center = (bbmin + bbmax) * 0.5
                coords = (coords - center)
                current_shift_and_scale[:3] = center
            if self.scale_factor is not None:
                coords_normalized = coords / self.scale_factor
                current_shift_and_scale[3] = self.scale_factor
            else:
                box_size = (bbmax - bbmin).max() + 1.0e-6
                current_shift_and_scale[3] = 1 / (2.0 * self.norm_range / box_size)
                coords_normalized = coords / current_shift_and_scale[3]
        else:
            # UNIT SPHERE NORMALIZATION:
            if self.zero_mean:
                centroid = torch.mean(coords, axis=0)
                coords = coords - centroid
                current_shift_and_scale[:3] = centroid
            if self.scale_factor is not None:
                max_distance = self.scale_factor
            else:
                # max_distance = torch.max(abs(coords_normalized)) / self.norm_range  ## INCORRECT, DOES NOT CONSIDER RADIAL DISTANCE
                max_distance = torch.max(torch.linalg.norm(coords, dim=1)) / self.norm_range
            coords_normalized = coords / max_distance        
            current_shift_and_scale[3] = max_distance
        
        if current_shift_and_scale[3] <= 0:
            raise ValueError("Invalid scaling factor")
        if self.return_shift_and_scale:
            return coords_normalized, current_shift_and_scale
        else:
            return coords_normalized

    @staticmethod
    def unnormalize(coords: torch.Tensor, shift_and_scale: torch.Tensor):
        """
        Undo normalization using shift and scale output from Normalize(). Note
        that this function will not verify that you have provided the correct
        shift and scale parameters for the given set of points.
        """
        assert coords.ndim == 2
        assert shift_and_scale.shape == (4,)
        if shift_and_scale[3] <= 0:
            raise ValueError("Invalid scaling factor")
        coords_unnormalized = coords * shift_and_scale[3] + shift_and_scale[:3]
        return coords_unnormalized
