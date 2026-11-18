import numpy as np 
import numpy.typing as npt
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors


def visualise_heightmap(heightmap_image: npt.NDArray[np.float32], rgb_image: npt.NDArray[np.uint8] = None):
    '''
    Allows for easy visualisation of ground heightmaps or aerial CHMv2 heightmaps 
    Inputs:
    - heightmap_image: Heightmap of size [HxW]
    - rgb_image (optional): RGB image of size [HxWx3].  If provided, will visualise the height overlaid on the RGB image
    Returns:
    - heightmap visualisation image: RGB image of size [HxWx3], with height visualised either by itself or superimposed over
      the corresponding RGB image 
    '''
    heightmap_image = heightmap_image / 1000. # Convert to metres
    no_depth_measurement_mask = heightmap_image == 0
    
    cmap = mpl.colormaps.get_cmap('gist_ncar_r')
    cmap.set_bad('black')
    norm = mcolors.Normalize(vmin=0., vmax=60.)
    
    norm_heightmap_image = norm(heightmap_image)
    heightmap_colorised = (cmap(norm_heightmap_image)[..., :3] * 255).astype(np.uint8)
    heightmap_colorised[no_depth_measurement_mask] = 0
    
    if rgb_image is not None:
        heightmap_colorised[no_depth_measurement_mask] = rgb_image[no_depth_measurement_mask] // 2
    
    return heightmap_colorised
