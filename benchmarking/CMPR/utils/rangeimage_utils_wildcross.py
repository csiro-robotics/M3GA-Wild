from typing import Union, Optional
import numpy as np
from PIL import Image
import numpy.typing as npt
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors

def range_projection(current_vertex: npt.NDArray, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range: Optional[float] = 5.0):
    """ Project a pointcloud into a spherical projection, range image.
    Args:
        current_vertex: raw point clouds
    Returns:
        proj_range: projected range image with depth, each pixel contains the corresponding depth
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        proj_intensity: each pixel contains the corresponding intensity
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
    """
    if max_range is None:
        max_range = np.inf
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]
    intensity = current_vertex[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    intensity = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1,
                        dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                        dtype=np.int32)  # [H,W] index (-1 is no data)
    proj_intensity = np.full((proj_H, proj_W), -1,
                            dtype=np.float32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices
    proj_intensity[proj_y, proj_x] = intensity
    # tensor
    # circular convolutions? remove overlap?
    return proj_range, proj_vertex, proj_intensity, proj_idx

def pointcloud_to_heightmap(
    points: npt.NDArray,
    extent: Union[float, tuple[float, float]],
    resolution: Union[int, tuple[int, int]],
    max_height: Optional[float] = None,
):
    """
    points: (N, 4) array of [x, y, z, int] in local coordinates.
    extent: (height_m, width_m) spatial coverage in metres. If a single value, square extent is assumed.
    resolution: (height_px, width_px) output resolution. If a single value, square resolution is assumed.
    max_height: maximum height value of heightmap.

    Returns:
        heightmap: (H, W) array with max height per cell, -1 for empty cells.
    """
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
    if max_height is not None:
        assert max_height > 0
        z = np.minimum(z, max_height)

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
    return heightmap

def normalise_range_image(image: npt.NDArray[np.float32], max_range: Optional[float] = 40., clip_invalid_depth: bool = False):
    """
    Normalise a (W, H) range image/heightmap and covert to RGB format, such that pixel ranges are
    normalised to uint8 range with specified max_range (uses per-image max if not specified). 
    """
    if clip_invalid_depth:
        image = image.clip(min=0)
    if max_range is None:
        max_range = float(image.max())
    range_scale_factor = 255 / max_range
    image = Image.fromarray(np.uint8(image * range_scale_factor) , 'L')
    image = np.stack((image,)*3, axis=-1)
    return image

def loadCloudFromBinary(file, cols=3):
    bin_pcd = np.fromfile(file, dtype=np.float32)
    points = bin_pcd.reshape(-1, 4)
    return points
    
def createRangeImage(points, crop=True, crop_distance=False, distance_threshold=100):
    if crop_distance:
        dis = np.linalg.norm(points[:, :3], axis=1)
        points = points[np.where(dis<distance_threshold)[0]]
    image,_,_,_ = range_projection(points)
    image = Image.fromarray(np.uint8(image * 5.1) , 'L')
    image = np.stack((image,)*3, axis=-1)
    if crop:
        clipStart = image.shape[1] // 3      
        image = image[0:32, clipStart:clipStart*2, :]
    return image

def createRangeImageWildVPR(points, crop=True, crop_distance=False, distance_threshold=100, max_range: Optional[float] = 50., idx=0):
    if crop_distance:
        dis = np.linalg.norm(points[:, :3], axis=1)
        points = points[np.where(dis<distance_threshold)[0]]
    image,_,_,_ = range_projection(points, fov_up=45.0, proj_H=160, max_range=max_range)  # DEFAULT FROM WildCross
    # image,_,_,_ = range_projection(points, fov_up=60.0, fov_down=-25.0, proj_H=210, max_range=max_range)

    image = normalise_range_image(image=image, max_range=max_range)
    if crop:
        clipStart = image.shape[1] // 3      
        image = image[:, clipStart:clipStart*2, :]

    return image

def createRangeImageM3GAWild(
    points,
    crop=True,
    crop_distance=False,
    distance_threshold=100,
    max_range: Optional[float] = 50.,
    source='_ground',
    clip_invalid_depth: bool = True,
    idx=0,
):
    if crop_distance:
        dis = np.linalg.norm(points[:, :3], axis=1)
        points = points[np.where(dis<distance_threshold)[0]]
    # image,_,_,_ = range_projection(points, fov_up=45.0, proj_H=160, max_range=max_range)  # DEFAULT FROM WildCross
    if source == '_ground':
        image,_,_,_ = range_projection(points, fov_up=60.0, fov_down=-25.0, proj_H=210, max_range=max_range)  # This config seems to capture more canopy
    elif source == '_aerial':
        # Shift z to start at roughly handheld scanner height from ground
        points_copy = points.copy()
        z_offset = 5  # metres
        points_copy[:, 2] = points_copy[:, 2] - points_copy[:, 2].min() - z_offset
        image,_,_,_ = range_projection(points_copy, fov_up=80.0, fov_down=-25.0, proj_W=896, proj_H=224, max_range=max_range)
    else:
        raise ValueError()

    image = normalise_range_image(image=image, max_range=max_range, clip_invalid_depth=clip_invalid_depth)
    if crop:
        clipStart = image.shape[1] // 3      
        image = image[:, clipStart:clipStart*2, :]

    return image

def createHeightMapM3GAWild(
    points,
    max_height: Optional[float] = 40.,
    extent=60.,
    resolution=(224,224),
    clip_invalid_depth: bool = True,
    idx=0,
):
    """
    points: (N, 4) array of point cloud
    max_height: (float) max height of heightmap in metres
    extent: (float, tuple[float, float]) y, x spatial coverage of heightmap
    resolution: (int, tuple[int, int]) H, W resolution of heightmap

    Returns:
        heightmap_image: (H, W, 3) array
    """
    
    image = pointcloud_to_heightmap(points, extent=extent, resolution=resolution, max_height=max_height)
    image = normalise_range_image(image=image, max_range=max_height, clip_invalid_depth=clip_invalid_depth)

    return image

def loadLidarImage(filename, seq, data_path):
    lidar_points = loadCloudFromBinary(f"{data_path}/{seq}/velodyne/{filename}.bin")
    lidar_image = createRangeImage(lidar_points)
    return lidar_image


def visualiseRangeImageWildVPR(
    range_image: npt.NDArray[np.float32],
    rgb_image: npt.NDArray[np.uint8] = None,
    max_range: float = 60,
    colormap='gist_ncar_r'
):
    '''
    Allows for easy visualisation of the range image 
    Inputs:
    - range_image: Range image of size [HxW]
    - rgb_image (optional): RGB image of size [HxWx3].  If provided, will visualise the depth overlaid on the RGB image
    - max_range (float): Max range of range image. Used for normalising colormap.
    Returns:
    - depth visualisation image: RGB image of size [HxWx3], with depth visualised either by itself or superimposed over
      the corresponding RGB image 
    '''
    #depth_image = depth_image / 1000. # Convert to metres
    no_depth_measurement_mask = range_image <= 0
    
    cmap = mpl.colormaps.get_cmap(colormap)
    cmap.set_bad('black')
    norm = mcolors.Normalize(vmin=0., vmax=max_range)
    
    norm_depth_image = norm(range_image)
    depth_colorised = (cmap(norm_depth_image)[..., :3] * 255).astype(np.uint8)
    depth_colorised[no_depth_measurement_mask] = 0
    
    if rgb_image is not None:
        depth_colorised[no_depth_measurement_mask] = rgb_image[no_depth_measurement_mask] // 2
    
    return depth_colorised