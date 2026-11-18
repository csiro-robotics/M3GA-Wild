import copy
import os
from typing import Optional, Union, Tuple

import torch
from torch import Tensor
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image

from misc.poses import apply_transform


def draw_registration_result(source, target, transformation):
    if not isinstance(source, o3d.geometry.PointCloud):
        source = make_open3d_point_cloud(source)
    if not isinstance(target, o3d.geometry.PointCloud):
        target = make_open3d_point_cloud(target)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def draw_pc(pc):
    if not isinstance(pc, o3d.geometry.PointCloud):
        pc = make_open3d_point_cloud(pc)
    pc = copy.deepcopy(pc)
    pc.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([pc],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def plot_points(points: np.ndarray, show=True, ax=None, point_size=None, cmap='viridis'):
    """
    Plots a point cloud using matplotlib. Colormap is based on z height.

    Args:
        points (ndarray): Point cloud of shape (N, 3), with (x,y,z) coords.
        show (bool): Show plot
        ax (matplotlib.axes): Matplotlib axes to plot on
        point_size (float): Size of points
        cmap (str): Matplotlib colormap to use

    Returns:
        Optional[fig]
    """    
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(9,8), layout='tight')
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    if point_size is not None:
        ax.scatter(*points.T, c=points.T[2], s=point_size, cmap=cmap)
    else:
        ax.scatter(*points.T, c=points.T[2], cmap=cmap)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal', adjustable='box')
    if show:
        plt.show()
    return fig

def plot_points_and_image(points: np.ndarray, image: Image, show=True, axs=None, point_size=None, cmap='viridis'):
    """
    Plots a point cloud and image using matplotlib. Colormap is based on z height.

    Args:
        points (ndarray): Point cloud of shape (N, 3), with (x,y,z) coords.
        image (Image): PIL Image.
        show (bool): Show plot
        axs (List[matplotlib.axes]): Matplotlib axes to plot on
        point_size (float): Size of points
        cmap (str): Matplotlib colormap to use

    Returns:
        Optional[fig]
    """
    fig = None
    if axs is None:
        axs = []
        fig = plt.figure(figsize=(12,5), layout='tight')
        axs.append(fig.add_subplot(1, 2, 1, projection='3d'))
        axs.append(fig.add_subplot(1, 2, 2))
    plot_points(points=points, show=show, ax=axs[0], point_size=point_size, cmap=cmap)
    axs[1].imshow(image)
    axs[1].set_axis_off()
    if show:
        plt.show()
    return fig

def plot_registration_result(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    show=True,
    point_size: float = 2,
    alpha: float = 0.3,
):
    """
    Plots registration between two point clouds using matplotlib. 

    Args:
        source (ndarray): Point cloud of shape (N, 3), with (x,y,z) coords.
        target (ndarray): Point cloud of shape (M, 3), with (x,y,z) coords.
        transform (ndarray): SE(3) transformation of shape (4,4) from src->tgt
    """    
    assert transform.shape == (4,4)
    source_colour = np.array([[1, 0.706, 0]])
    target_colour = np.array([[0, 0.651, 0.929]])
    source_temp = apply_transform(source, transform)
    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(*source_temp.T, c=source_colour, s=point_size, alpha=alpha)
    ax.scatter(*target.T, c=target_colour, s=point_size, alpha=alpha)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal', adjustable='box')
    if show:
        plt.show()


def icp(
    anchor_pc,
    positive_pc,
    transform: Optional[np.ndarray] = None,
    point2plane=False,
    gicp=False,
    inlier_dist_threshold: float = 1.2,
    max_iteration: int = 200,
    voxel_size: Optional[float] = 0.1,
):
    assert not(point2plane and gicp), "Choose either point2plane or gicp method, not both"
    # transform: initial alignment transform
    if transform is not None:
        transform = transform.astype(float)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(anchor_pc)
    if voxel_size is not None:
        pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(positive_pc)
    if voxel_size is not None:
        pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    if gicp:
        transform_estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
        registration_fn = o3d.pipelines.registration.registration_generalized_icp
    else:
        if point2plane:
            pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
            pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
            transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        registration_fn = o3d.pipelines.registration.registration_icp

    if transform is not None:
        reg_p2p = registration_fn(pcd1, pcd2, inlier_dist_threshold, transform,
                                  estimation_method=transform_estimation,
                                  criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    else:
        reg_p2p = registration_fn(pcd1, pcd2, inlier_dist_threshold,
                                  estimation_method=transform_estimation,
                                  criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

    return reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse


def two_stage_icp(
    anchor_pc,
    positive_pc,
    transform: Optional[np.ndarray] = None,
    point2plane=False,
    gicp=False,
    inlier_dist_threshold: float = 1.2,
    max_iteration: int = 200,
    voxel_size: Optional[float] = 0.1,
    two_stage_inlier_dist_threshold: float = 5.5,
    two_stage_max_iteration: int = 50,
    two_stage_voxel_size: Optional[float] = 0.5,
):
    """
    Compute a two-stage ICP refinement, from coarse-to-fine. Used to fix drift
    on DCC sequence, which is not fixable in all situations with single-stage ICP.
    """
    assert not(point2plane and gicp), "Choose either point2plane or gicp method, not both"

    transform_coarse, _, _ = icp(
        anchor_pc=anchor_pc,
        positive_pc=positive_pc,
        transform=transform,
        inlier_dist_threshold=two_stage_inlier_dist_threshold,
        max_iteration=two_stage_max_iteration,
        voxel_size=two_stage_voxel_size,
    )
    transform_fine, fitness, rmse = icp(
        anchor_pc=anchor_pc,
        positive_pc=positive_pc,
        transform=transform_coarse,
        point2plane=point2plane,
        gicp=gicp,
        inlier_dist_threshold=inlier_dist_threshold,
        max_iteration=max_iteration,
        voxel_size=voxel_size,
    )
    return transform_fine, fitness, rmse


def fast_global_registration(
    anchor_pc,
    positive_pc,
    inlier_dist_threshold: float = 0.2,
    max_iteration: int = 64,
    voxel_size: Optional[float] = 0.4,
):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(anchor_pc)
    if voxel_size is not None:
        pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(positive_pc)
    if voxel_size is not None:
        pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    pcd1_fpfh = compute_fpfh(pcd1, voxel_size=voxel_size)
    pcd2_fpfh = compute_fpfh(pcd2, voxel_size=voxel_size)
    
    # Recommended threshold is 0.5x voxel size
    fgr_option = o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=inlier_dist_threshold,
            iteration_number=max_iteration,
    )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pcd1, pcd2, pcd1_fpfh, pcd2_fpfh, option=fgr_option,
    )

    return result.transformation, result.fitness, result.inlier_rmse


def compute_fpfh(pcd, voxel_size: Optional[float] = 0.4):
    # Radius normal and feature size based on Open3D recommendations:
    # https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#Extract-geometric-feature 
    if voxel_size is None:
        voxel_size = 0.4
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_fpfh


def make_open3d_feature(data, dim=None, npts=None):
    feature = o3d.pipelines.registration.Feature()
    if not(dim is None or npts is None):
        feature.resize(dim, npts)
    feature.data = data.cpu().numpy().astype('d').transpose()
    return feature


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def registration_with_ransac_from_feats(
    src_points,
    ref_points,
    src_feats,
    ref_feats,
    distance_threshold=0.05,
    ransac_n=3,
    num_iterations=50000,
    confidence=0.999,
):
    r"""
    Compute the transformation matrix from src_points to ref_points
    """
    src_pcd = make_open3d_point_cloud(src_points)
    ref_pcd = make_open3d_point_cloud(ref_points)
    src_feats = make_open3d_feature(src_feats)
    ref_feats = make_open3d_feature(ref_feats)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pcd,
        ref_pcd,
        src_feats,
        ref_feats,
        distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(num_iterations, confidence),
    )

    return result.transformation


def registration_with_ransac_from_correspondences(
    src_points,
    ref_points,
    correspondences=None,
    distance_threshold=0.05,
    ransac_n=3,
    num_iterations=10000,
    confidence=0.999,
):
    r"""
    Compute the transformation matrix from src_points to ref_points
    """
    src_pcd = make_open3d_point_cloud(src_points)
    ref_pcd = make_open3d_point_cloud(ref_points)

    if correspondences is None:
        indices = np.arange(src_points.shape[0])
        correspondences = np.stack([indices, indices], axis=1)
    correspondences = o3d.utility.Vector2iVector(correspondences)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_pcd,
        ref_pcd,
        correspondences,
        distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(num_iterations, confidence),
    )

    return result.transformation


def preprocess_pointcloud(
    pc,
    remove_zero_points: bool = False,
    min_x: float = None,
    max_x: float = None,
    min_y: float = None,
    max_y: float = None,
    min_z: float = None,
    max_z: float = None,
):
    if remove_zero_points:
        mask = np.all(np.isclose(pc, 0.), axis=1)
        pc = pc[~mask]
    if min_x is not None:
        mask = pc[:, 0] > min_x
        pc = pc[mask]
    if max_x is not None:
        mask = pc[:, 0] <= max_x
        pc = pc[mask]
    if min_y is not None:
        mask = pc[:, 1] > min_y
        pc = pc[mask]
    if max_y is not None:
        mask = pc[:, 1] <= max_y
        pc = pc[mask]
    if min_z is not None:
        mask = pc[:, 2] > min_z
        pc = pc[mask]
    if max_z is not None:
        mask = pc[:, 2] <= max_z
        pc = pc[mask]
    return pc


def hash_points(points: torch.Tensor, voxel_size: float = 0.3, return_inverse=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    # Quantize coordinates
    coords = torch.floor(points / voxel_size).long()
    # Hash voxel coordinates (using generic primes)
    primes = torch.tensor([73856093, 19349663, 83492791], device=points.device)
    hashed = (coords * primes).sum(dim=-1)
    return torch.unique(hashed, return_inverse=return_inverse)


def compute_min_overlap(pc1: torch.Tensor, pc2: torch.Tensor, voxel_size: float = 0.3):
    """
    Compute the minimum overlap between two point clouds.

    Overlap is defined as: |intersection(voxels(A), voxels(B))| / max(|voxels(A)|, |voxels(B)|)
    
    This gives ~50% when one cloud is the other with half its volume removed, regardless of which
    direction you measure.

    Note hash collisions may be possible for small voxel sizes, but are rare enough to ignore.

    Args:
        pc1: (N, 3) float tensor
        pc2: (M, 3) float tensor
        voxel_size: edge length of each voxel (in same units as point coords)

    Returns:
        overlap in [0, 1]
    """
    vox1 = hash_points(pc1, voxel_size)
    vox2 = hash_points(pc2, voxel_size)
    # Intersection via set logic
    all_vox = torch.cat([vox1, vox2])
    unique, counts = torch.unique(all_vox, return_counts=True)
    intersection = (counts > 1).sum().item()
    # return min overlap
    return intersection / max(len(vox1), len(vox2))


def pointcloud_to_heightmap(
    points: np.ndarray,
    extent: Union[float, tuple[float, float]],
    resolution: Union[int, tuple[int, int]],
    max_height: Optional[float] = None,
    normalise_ground: bool = False,
    median_ksize: Optional[int] = 3,
):
    """
    Args:
        points: (N, 4) array [x, y, z, int]
        extent: (H, W) spatial coverage in metres
        resolution: (H, W) output resolution
        max_height: clamp max height
        normalise_ground (bool): 
            Normalise ground height by subtracting CSF ground height from canopy height (ensures ground is consistently 0m). 
            May behave weirdly if ground is sparsely sampled.
        median_ksize (int): median filter kernel size

    Returns:
        heightmap: (H, W) height above CSF ground
    """
    import CSF
    from scipy.ndimage import distance_transform_edt, median_filter

    if isinstance(extent, float):
        extent = (extent, extent)
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    extent_h, extent_w = extent
    H, W = resolution

    half_h = extent_h / 2.0
    half_w = extent_w / 2.0

    # Keep points within map bounds
    mask = (
        (points[:, 0] >= -half_w) & (points[:, 0] < half_w) &
        (points[:, 1] >= -half_h) & (points[:, 1] < half_h)
    )
    pts = points[mask]

    z = pts[:, 2]
    # Shift z to start at 0
    z = z - z.min()
    pts[:, 2] = z

    if not normalise_ground:
        # Convert XY to pixel indices
        cols = ((pts[:, 0] + half_w) * W / extent_w).astype(np.int32)
        rows = ((half_h - pts[:, 1]) * H / extent_h).astype(np.int32)

        valid = (
            (rows >= 0) & (rows < H) &
            (cols >= 0) & (cols < W)
        )
        rows, cols, z = rows[valid], cols[valid], z[valid]

        heightmap = np.full((H, W), -1.0, dtype=np.float32)
        np.maximum.at(heightmap, (rows, cols), z)

        # Clamp
        if max_height is not None:
            heightmap = np.minimum(heightmap, max_height)
        return heightmap

    # -------------------------
    # 1. CSF ground extraction
    # -------------------------
    csf = CSF.CSF()

    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = 0.5
    csf.params.class_threshold = 0.5
    csf.params.rigidness = 3

    csf.setPointCloud(pts[:, :3].astype(np.float64))

    ground_idx = CSF.VecInt()       # index of ground points
    non_ground_idx = CSF.VecInt()   # index of non-ground points
    csf.do_filtering(ground_idx, non_ground_idx)
    ground_idx = np.array(ground_idx)
    non_ground_idx = np.array(non_ground_idx)

    # If CSF fails or no ground, fallback
    if len(ground_idx) == 0:
        print('No ground pts found')
        ground_pts = pts
    else:
        ground_pts = pts[ground_idx]

    # -------------------------
    # 2. Raster Helper
    # -------------------------
    def raster_max(p):
        cols = ((p[:, 0] + half_w) * W / extent_w).astype(np.int32)
        rows = ((half_h - p[:, 1]) * H / extent_h).astype(np.int32)

        valid = (
            (rows >= 0) & (rows < H) &
            (cols >= 0) & (cols < W)
        )
        rows, cols, zvals = rows[valid], cols[valid], p[valid, 2]

        dsm = np.full((H, W), -1.0, dtype=np.float32)
        np.maximum.at(dsm, (rows, cols), zvals)
        return dsm

    # -------------------------
    # 3. Rasterise DSM/DTM from pts and CSF ground points
    # -------------------------
    dsm = raster_max(pts)
    dtm = raster_max(ground_pts)

    # Fill missing ground via simple interpolation (optional but helpful)
    valid = dtm > -1
    if np.any(~valid):
        mask = valid.astype(np.uint8)
        _, (iy, ix) = distance_transform_edt(1 - mask, return_indices=True)
        dtm = dtm[iy, ix]
    
    # Median filtering for smoothing
    if median_ksize is not None and median_ksize > 1:
        dsm = median_filter(dsm, size=median_ksize)
        dtm = median_filter(dtm, size=median_ksize)

    # -------------------------
    # 4. Height above ground
    # -------------------------
    heightmap = dsm - dtm

    # Clamp
    if max_height is not None:
        heightmap = np.minimum(heightmap, max_height)

    heightmap[dsm < 0] = -1.0
    return heightmap


def heightmap_to_pointcloud(heightmap: np.ndarray, gsd: float = 0.075):
    """
    Args:
        heightmap: (H, W) float32 array, height in metres (-1 = invalid)
        gsd: pixel size in metres

    Returns:
        points: (N, 3) array [x, y, z]
    """
    H, W = heightmap.shape

    # Pixel grid (centered at origin)
    xs = (np.arange(W) - W / 2 + 0.5) * gsd
    ys = (H / 2 - np.arange(H) - 0.5) * gsd

    xx, yy = np.meshgrid(xs, ys)

    z = heightmap

    valid = np.isfinite(z) & (z > -1)

    x = xx[valid]
    y = yy[valid]
    z = z[valid]

    return np.stack([x, y, z], axis=1)


class PointCloudLoader:
    def __init__(self):
        # remove_zero_points: remove points with all zero coordinates
        # remove_ground_plane: remove points on ground plane level and below
        # ground_plane_level: ground plane level
        self.remove_zero_points = True
        self.remove_ground_plane = True
        self.ground_plane_level = None
        self.max_range = None
        self.set_properties()

    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level. Must be defined in inherited classes.
        raise NotImplementedError('set_properties must be defined in inherited classes')

    def __call__(self, file_pathname):
        # Reads the point cloud from a disk and preprocess (optional removal of zero points and points on the ground
        # plane and below
        # file_pathname: relative file path
        assert os.path.exists(file_pathname), f"Cannot open point cloud: {file_pathname}"
        pc = self.read_pc(file_pathname)
        assert pc.shape[1] == 3

        if self.max_range is not None:
            pc = preprocess_pointcloud(pc, min_x=-self.max_range, max_x=self.max_range,
                                       min_y=-self.max_range, max_y=self.max_range)
        
        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]

        if self.remove_ground_plane:
            mask = pc[:, 2] > self.ground_plane_level
            pc = pc[mask]

        return pc

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        raise NotImplementedError("read_pc must be overloaded in an inheriting class")