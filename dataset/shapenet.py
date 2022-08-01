'''
Author: Jinguang Tong
Affliction: Australia National University, DATA61 CSIRO
'''

import os
import glob
import numpy as np

from skimage import io
from copy import copy
from graphics import Voxelgrid
from scipy.ndimage.morphology import binary_dilation
from torch.utils.data import Dataset
from utils.data import add_kinect_noise, add_depth_noise, add_outliers
from dataset.binvox_utils import read_as_3d_array


class ShapeNet(Dataset):

    def __init__(self, config):
        self.root_dir = os.path.expanduser(config.root_dir)
        self.resolution = (config.resy, config.resx)
        self.xscale = self.resolution[0] / 480
        self.yscale = self.resolution[1] / 640

        self.transform = config.transform

        self.scene_list = config.scene_list
        self.noise_scale = config.noise_scale
        self.outlier_scale = config.outlier_scale
        self.outlier_fraction = config.outlier_fraction

        self.grid_resolution = config.grid_resolution

        self._load_frames()

    def _load_frames(self):

        self.frames = []
        self._scenes = []

        with open(self.scene_list, 'r') as file:

            for line in file:
                scene, obj = line.rstrip().split('\t')

                self._scenes.append(os.path.join(scene, obj))

                path = os.path.join(self.root_dir, scene, obj, 'data', '*.depth.png')

                files = glob.glob(path)

                self.frames.extend(f.replace('.depth.png', '') for f in files)

    @property
    def scenes(self):
        return self._scenes

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):

        frame = self.frames[item]

        pathsplit = frame.split(os.path.sep)
        sc = pathsplit[-4]
        obj = pathsplit[-3]
        scene_id = f'{sc}/{obj}'
        frame_id = pathsplit[-1]
        frame_id = int(frame_id)
        depth = io.imread(f'{frame}.depth.png')
        depth = depth.astype(np.float32)
        depth = depth / 1000.

        step_x = depth.shape[0] / self.resolution[0]
        step_y = depth.shape[1] / self.resolution[1]

        index_y = [int(step_y * i) for i in range(int(depth.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in range(int(depth.shape[0] / step_x))]

        depth = depth[:, index_y]
        depth = depth[index_x, :]

        mask = copy(depth)
        mask[mask == np.max(depth)] = 0
        mask[mask != 0] = 1
        sample = {'frame_id': frame_id, 'mask': copy(mask)}
        gradient_mask = binary_dilation(mask, iterations=5)
        mask = binary_dilation(mask, iterations=8)
        sample['routing_mask'] = mask
        sample['gradient_mask'] = gradient_mask

        depth[mask == 0] = 0

        sample['depth'] = depth
        sample['noisy_depth'] = add_kinect_noise(copy(depth), sigma_fraction=self.noise_scale)
        sample['noisy_depth_octnetfusion'] = add_depth_noise(copy(depth), noise_sigma=self.noise_scale, seed=42)
        sample['outlier_depth'] = add_outliers(copy(sample['noisy_depth_octnetfusion']), scale=self.outlier_scale,
            fraction=self.outlier_fraction)

        intrinsics = np.loadtxt(f'{frame}.intrinsics.txt')
        # adapt intrinsics to camera resolution
        scaling = np.eye(3)
        scaling[1, 1] = self.yscale
        scaling[0, 0] = self.xscale

        sample['intrinsics'] = np.dot(scaling, intrinsics)

        extrinsics = np.loadtxt(f'{frame}.extrinsics.txt')
        extrinsics = np.linalg.inv(extrinsics)
        sample['extrinsics'] = extrinsics

        sample['scene_id'] = scene_id

        for key, value in sample.items():
            if type(value) is not np.ndarray and type(sample[key]) is not str:
                sample[key] = np.asarray(sample[key])

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_grid_occ(self, scene, truncation=None):

        sc, obj = scene.split(os.path.sep)

        if self.grid_resolution == 256:
            filepath = os.path.join(self.root_dir, sc, obj, 'voxels', '*.binvox')
        else:
            filepath = os.path.join(self.root_dir, sc, obj, 'voxels', f'*.{self.grid_resolution}.binvox')
        filepath = glob.glob(filepath)[0]

        with open(filepath, 'rb') as file:
            volume = read_as_3d_array(file)

        resolution = 1. / self.grid_resolution

        grid = Voxelgrid(resolution)
        occ = Voxelgrid(resolution)
        bbox = np.zeros((3, 2))
        bbox[:, 0] = volume.translate
        bbox[:, 1] = bbox[:, 0] + resolution * volume.dims[0]

        grid.from_array(volume.data.astype(np.int), bbox)
        occ.from_array(volume.data.astype(np.float), bbox)
        # calculate tsdf
        grid.transform()
        grid.volume *= resolution

        if truncation is not None:
            grid.volume[np.abs(grid.volume) >= truncation] = truncation
        return grid, occ

    def get_occ(self, scene):
        # TODO
        sc, obj = scene.split(os.sep)
        if self.grid_resolution == 256:
            filepath = os.path.join(self.root_dir, sc, obj, 'occupancy', '*.npz')
        else:
            filepath = os.path.join(self.root_dir, sc, obj, 'occupancy', f'*.{self.grid_resolution}.npz')

        filepath = glob.glob(filepath)[0]
        volume = np.load(filepath)

        occupancies = volume['occupancies']

        resolution = 1. / self.grid_resolution
        return Voxelgrid(resolution)

        

if __name__ == '__main__':
    from utils.loading import load_config_from_yaml

    config = load_config_from_yaml('configs/fusion/shapenet.noise.005.without.routing.yaml')
    config.DATA.scene_list = config.DATA.train_scene_list
    dataset = ShapeNet(config.DATA)
    dataset.get_grid('04530566/10e10b663a81801148c1c53e2c827229')
