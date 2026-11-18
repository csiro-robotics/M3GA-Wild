import numpy as np 
import numpy.typing as npt
import PIL.Image as Image 
import cv2 

def load_submap(submap_path: str, intensity=False) -> npt.NDArray[np.float32]:
    '''
    Loads a given submap from the disk
    Inputs:
    - submap_path: String containing location of the submap on disk
    - drop_intensity: Boolean which when false drops the intensity channel and only returns XYZ coordinates of the points
    Outputs:
    - submap: numpy array of dimension [N x 3/4], where N is the number of points and the four channels represent
              the X, Y, Z coordinates and LiDAR intensity of each points
    
    '''
    submap = np.fromfile(submap_path, dtype=np.float32)
    submap = submap.reshape(-1,4)
    if intensity == False:
        submap = submap[:,:3]
    return submap 

def load_rgb_image(image_path: str) -> npt.NDArray[np.uint8]:
    '''
    Loads a given RGB image from the disk
    Inputs:
    - image_path: String containing path to the image file on disk
    Outputs:
    - image_array: 3 channel numpy array of dimension [H x W x 3], representing the RGB image.
    '''
    image_array = np.array(Image.open(image_path))
    return image_array

def load_heightmap(heightmap_path: str) -> npt.NDArray[np.uint16]:
    '''
    Loads a given heightmap from the disk
    Inputs:
    - heightmap_path: String containing path to the heightmap file on disk
    Outputs:
    - heightmap_mm: Numpy array of dimension [H x W], containing the heightmap associated with each pixel in mm
    '''
    heightmap_mm = cv2.imread(heightmap_path, cv2.IMREAD_UNCHANGED)
    return heightmap_mm 
