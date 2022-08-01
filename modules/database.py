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

        self.scenes_tsdf = {}
        self.scenes_occ = {}
        self.feature_est = {}
        self.tsdf_est = {}
        self.occ_est = {}
        self.fusion_weights = {}
        self.update_counts = {}

        for s in dataset.scenes:

            grid, occ = dataset.get_grid_occ(s, truncation=self.initial_value)


            self.scenes_tsdf[s] = grid
            # TODO get occupancy volume from input data
            self.scenes_occ[s] = occ
            # self.scenes_occ[s] = dataset.get_occ(s)

            # init_volume = self.initial_value * np.ones_like(grid.volume)
            init_feature = self.initial_value * np.ones(
                grid.volume.shape + (config.len_feature,)
            )

            init_tsdf = self.initial_value * np.ones(grid.volume.shape)
            init_occ = np.zeros(grid.volume.shape)

            self.feature_est[s] = Voxelgrid(self.scenes_tsdf[s].resolution)
            self.feature_est[s].from_array(init_feature, self.scenes_tsdf[s].bbox)
            self.tsdf_est[s] = Voxelgrid(self.scenes_tsdf[s].resolution)
            self.tsdf_est[s].from_array(init_tsdf, self.scenes_tsdf[s].bbox)
            self.occ_est[s] = Voxelgrid(self.scenes_tsdf[s].resolution)
            self.occ_est[s].from_array(init_occ, self.scenes_tsdf[s].bbox)
            self.fusion_weights[s] = np.zeros(self.scenes_tsdf[s].volume.shape)
            self.update_counts[s] = np.zeros(self.scenes_tsdf[s].volume.shape)            
            
    def __getitem__(self, item):

        sample = {
            'occ': self.scenes_occ[item].volume,
            'tsdf': self.scenes_tsdf[item].volume,
            'current': self.feature_est[item].volume,
            'origin': self.scenes_tsdf[item].origin,
            'resolution': self.scenes_tsdf[item].resolution,
            'weights': self.fusion_weights[item],
            'counts': self.update_counts[item],
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    
    def __len__(self):
        return len(self.scenes_gt)
    
    def filter(self, value=2.):
        
        for key in self.feature_est.keys():
            weights = self.fusion_weights[key]
            self.feature_est[key].volume[weights < value] = self.initial_value
            self.fusion_weights[key][weights < value] = 0
            
    def save_to_workspace(self, workspace):

        for key in self.feature_est.keys():

            tsdf_volume = self.tsdf_est[key].volume
            occ_volume = self.occ_est[key].volume

            tsdf_file = key.replace(os.path.sep, '.') + '.tsdf.hf5'
            occ_file = key.replace(os.path.sep, '.') + '.weights.hf5'

            workspace.save_tsdf_data(tsdf_file, tsdf_volume)
            workspace.save_occ_data(occ_file, occ_volume)          
            
    
    def save(self, path, scene_id=None, epoch=None, groundtruth=False):
        if scene_id is None:
            raise NotImplementedError

    def evaluate(self, mode='train', workspace=None):

        eval_tsdf = {}

        for scene_id in self.feature_est.keys():

            if workspace is None:
                print("Evaluating ", scene_id, "...")
            else:
                workspace.log(f"Evaluating {scene_id} ...", mode)

            update_counts = self.update_counts[scene_id]
            tsdf_est = self.tsdf_est[scene_id].volume
            tsdf_gt = self.scenes_tsdf[scene_id].volume

            mask = np.copy(update_counts)
            mask[mask > 0] = 1.

            eval_scene_tsdf = evaluation(tsdf_est, tsdf_gt, mask)
            # eval_results_occ = evaluation()

            for key in eval_scene_tsdf.keys():
                if workspace is None:
                    print(key, eval_scene_tsdf[key])
                else:
                    workspace.log(f'{key} {eval_scene_tsdf[key]}', mode)

                if not eval_tsdf.get(key):
                    eval_tsdf[key] = eval_scene_tsdf[key]
                else:
                    eval_tsdf[key] += eval_scene_tsdf[key]

        # normalizing metrics
        for key in eval_tsdf:
            eval_tsdf[key] /= len(self.scenes_tsdf.keys())

        return eval_tsdf
             
    
    def reset(self):
        for scene_id in self.feature_est.keys():
            self.feature_est[scene_id].volume = self.initial_value * np.ones(self.feature_est[scene_id].volume.shape)
            self.fusion_weights[scene_id] = np.zeros(self.scenes_tsdf[scene_id].volume.shape)
            self.update_counts[scene_id] = np.zeros(self.scenes_tsdf[scene_id].volume.shape)
            

if __name__ == '__main__':
    v = Voxelgrid()
    print(v)