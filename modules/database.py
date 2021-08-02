import os
import h5py
import torch
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
        self.scenes_tsdf = {}
        self.scenes_occ = {}
        self.fusion_weights = {}
        self.update_counts = {}
        
        for s in dataset.scenes:
            
            grid = dataset.get_grid(s, truncation=self.initial_value)
            
            
            self.scenes_tsdf[s] = grid
            # TODO get occupancy volume from input data
            self.scenes_occ[s] = dataset.get_occ(s)
            
            # init_volume = self.initial_value * np.ones_like(grid.volume)
            init_volume = self.initial_value * np.ones(grid.volume.shape + tuple([config.len_feature]))
            
            self.scenes_est[s] = Voxelgrid(self.scenes_tsdf[s].resolution)
            self.scenes_est[s].from_array(init_volume, self.scenes_tsdf[s].bbox)
            self.fusion_weights[s] = np.zeros(self.scenes_tsdf[s].volume.shape)
            self.update_counts[s] = np.zeros(self.scenes_tsdf[s].volume.shape)

    def __getitem__(self, item):

        sample = dict()
        sample['occ'] = self.scenes_occ[item].volume
        sample['tsdf'] = self.scenes_tsdf[item].volume
        sample['current'] = self.scenes_est[item].volume
        sample['origin'] = self.scenes_tsdf[item].origin
        sample['resolution'] = self.scenes_tsdf[item].resolution
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
            
    def save_to_workspace(self, workspace):
        
        for key in self.scenes_est.keys():
            
            tsdf_volume = self.scenes_tsdf[key].volume
            occ_volume = self.scenes_tsdf[key].volume
            
            tsdf_file = key.replace(os.path.sep, '.') + '.tsdf.hf5'
            occ_file = key.replace(os.path.sep, '.') + '.weights.hf5'
            
            workspace.save_tsdf_data(tsdf_file, tsdf_volume)
            workspace.save_occ_data(occ_file, occ_volume)          
            
                    
    
    def reset(self):
        for scene_id in self.scenes_est.keys():
            self.scenes_est[scene_id].volume = self.initial_value * np.ones(self.scenes_est[scene_id].volume.shape)
            self.fusion_weights[scene_id] = np.zeros(self.scenes_tsdf[scene_id].volume.shape)
            self.update_counts[scene_id] = np.zeros(self.scenes_tsdf[scene_id].volume.shape)
            

if __name__ == '__main__':
    v = Voxelgrid()
    print(v)