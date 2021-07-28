import os
import h5py

import numpy as np

from torch.utils.data import Dataset
from graphics import Voxelgrid

from utils.metrics import evaluation

class Database(Dataset):
    
    def __init__(self, dataset, config):
        super().__init__()
        
        self.transform = config.transform
        self.initial_value = config.init_value
        
        self.scenes_gt = {}
        self.scenes_est = {}
        self.fusion_weights = {}
        
        for s in dataset.scenes:
            
            grid = dataset.get_grid(s, truncation=self.initial_value)
            
            self.scenes_gt[s] = grid
            
            init_volume = self.initial_value * np.ones_like(grid.volume)
            
            self.scenes_est[s] = Voxelgrid(self.scenes_gt[s].resolution)
            self.scenes_est[s].from_array(init_volume, self.scenes_gt[s].bbox)
            self.fusion_weights[s] = np.zeros(self.scenes_gt[s].volume.shape)
            self.update_counts[s] = np.zeros(self.scenes_gt[s].volume.shape)

    def __getitem__(self, item):

        sample = dict()
        
        sample['gt'] = self.scenes_gt[item].volume
        sample['current'] = self.scenes_est[item].volume
        sample['origin'] = self.scenes_gt[item].origin
        sample['resolution'] = self.scenes_gt[item].resolution
        sample['weights'] = self.fusion_weights[item]
        sample['counts'] = self.update_counts[item]

        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        return len(self.scenes_gt)
    
    def filter(self, value=2.):
        
        for key in self.scenes_est.keys():
            weights = self.fusion_weights[key]
            self.scenes_est[key].volume[weights < value] = self.initial_value
            self.fusion_weights[key][weights < value] = 0

if __name__ == '__main__':
    v = Voxelgrid()
    print(v)