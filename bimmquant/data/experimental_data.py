############################################################################################
## This script contains functionality to load the dataset SOCPristine, stored as tiff slices.
#############################################################################################


import numpy as np
from skimage import io
import os
import torch
from torch.utils.data import Dataset

def tiff_slices_to_npy(directory):
    '''Loads all tiff images in file directory into 3D nympy array'''

    data = []

    for ind, filename in enumerate(sorted(os.listdir(directory))):
        if filename.endswith(".tiff"):
            image = io.imread(os.path.join(directory,filename))
            data.append(image)

    return np.array(data).T

def normalize(arr, amin=0, amax=1):
    arr01=(arr-np.min(arr))/(np.max(arr)-np.min(arr))
    return arr01*(amax-amin)+amin

def gradientMagnitude3D(vol):
    ''' Computes 3D gradient magnitude through central differences'''
    gradx, grady, gradz = np.gradient(vol)
    return np.sqrt(gradx**2 + grady**2+ gradz**2)


class SOCPristineFor2Dmodel(Dataset):
    '''Prepares data so that pytorch dataloader functionality can be utilized. Edges are removed (1 voxel thick layer) since gradients can not be calculated with central differences there.
    tiffs_path: path to folder with tiff slices
    self.u: intensity tensor
    self.v: gradient tensor

    1. Loads all tiffs into a 3D numpy array
    2. Normalizes intensity values to the range [0, 1]
    3. Calculates gradient magnitudes
    4. Stores voxel intensities and gradient magnitudes as flat tensors (float64)
    '''

    def __init__(self, tiffs_path):
        data_vol = tiff_slices_to_npy(tiffs_path) #converts tiff images to numpy 3D array
        self.original_range = [np.min(data_vol), np.max(data_vol)]
        self.shape = [np.shape(data_vol)[0]-2, np.shape(data_vol)[1]-2, np.shape(data_vol)[2]-2] #edges are removed
        self.u, self.v = self.get_intensity_and_gradient_arrays(data_vol)

    def get_intensity_and_gradient_arrays(self, data_vol):
        data_vol_norm = normalize(data_vol.astype(np.float64)) #normalize into range 0, 1
        data_gradient = gradientMagnitude3D(data_vol_norm)
        u_data = torch.tensor(data_vol_norm[1:-1, 1:-1, 1:-1].ravel()) #remove edges since these are not calc. with central differences
        v_data =  torch.tensor(data_gradient[1:-1, 1:-1, 1:-1].ravel())
        return u_data, v_data

    def __getitem__(self, index):
        return self.u[index].float(), self.v[index].float()

    def __len__(self):
        return len(self.u)

class SOCPristineFor1Dmodel(Dataset):
    '''Prepares data so that pytorch dataloader functionality can be utilized. Edges are removed to match data import for 2D models (intensity + gradient magnitude)
    tiffs_path: path to folder with tiff slices
    self.u: intensity tensor

    1. Loads all tiffs into a 3D numpy array
    2. Normalizes intensity values to the range [0, 1]
    3. Stores voxel intensities as a flat tensor (float64)
    '''

    def __init__(self, tiffs_path):
        data_vol = tiff_slices_to_npy(tiffs_path) #converts tiff images to numpy 3D array
        self.original_range = [np.min(data_vol), np.max(data_vol)]
        self.shape = [np.shape(data_vol)[0]-2, np.shape(data_vol)[1]-2, np.shape(data_vol)[2]-2] #edges are removed
        self.u = self.get_intensity_array(data_vol)

    def get_intensity_array(self, data_vol):
        data_vol_norm = normalize(data_vol.astype(np.float64)) #normalize into range 0, 1
        u_data = torch.tensor(data_vol_norm[1:-1, 1:-1, 1:-1].ravel()) #remove edges: just to match dataset for 2D model
        return u_data

    def __getitem__(self, index):
        return self.u[index].float()

    def __len__(self):
        return len(self.u)
