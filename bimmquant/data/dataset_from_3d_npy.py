############################################################################################
## This script contains functionality to load any dataset stored as a numpy array (.npy)
#############################################################################################

import numpy as np
import torch
from torch.utils.data import Dataset



def gradientMagnitude3D(vol):
    ''' Computes 3D gradient magnitude through central differences'''
    gradx, grady, gradz = np.gradient(vol)
    return np.sqrt(gradx**2 + grady**2+ gradz**2)


class DatasetFor2Dmodel(Dataset):
    '''Prepares data so that pytorch dataloader functionality can be utilized. Edges are removed (1 voxel thick layer) since gradients can not be calculated with central differences there. No intensity normalization.
    data_vol_path: path to data (numpy array)
    self.u: intensity tensor
    self.v: gradient tensor

    1. Loads the 3D numpy array
    2. Calculates gradient magnitudes
    3. Stores voxel intensities and gradient magnitudes as flat tensors (float64)
    '''

    def __init__(self, data_vol_path):
        data_vol = np.load(data_vol_path)
        self.shape = [np.shape(data_vol)[0]-2, np.shape(data_vol)[1]-2, np.shape(data_vol)[2]-2] #edges are removed
        self.u, self.v = self.get_intensity_and_gradient_arrays(data_vol)

    def get_intensity_and_gradient_arrays(self, data_vol):
        data_gradient = gradientMagnitude3D(data_vol.astype(np.float64))
        u_data = torch.tensor(data_vol.astype(np.float64)[1:-1, 1:-1, 1:-1].ravel())
        v_data =  torch.tensor(data_gradient[1:-1, 1:-1, 1:-1].ravel()) #remove edges since these are not calc. with central differences
        return u_data, v_data

    def __getitem__(self, index):
        return self.u[index].float(), self.v[index].float()

    def __len__(self):
        return len(self.u)


class DatasetFor1Dmodel(Dataset):
    '''Prepares data so that pytorch dataloader functionality can be utilized. Edges are removed to match data import for 2D models (intensity + gradient magnitude). No intensity normalization.
    data_vol: numpy array
    self.u: intensity tensor

    1. Loads the 3D numpy array
    3. Stores voxel intensities as a flat tensors (float64)
    '''

    def __init__(self, data_vol):
        self.shape = [np.shape(data_vol)[0]-2, np.shape(data_vol)[1]-2, np.shape(data_vol)[2]-2] #edges are removed
        self.u = self.get_intensity_array(data_vol)

    def get_intensity_array(self, data_vol):
        u_data = torch.tensor(data_vol.astype(np.float64)[1:-1, 1:-1, 1:-1].ravel())
        return u_data

    def __getitem__(self, index):
        return self.u[index].float()

    def __len__(self):
        return len(self.u)
