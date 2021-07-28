'''
Description: 
Author: Jinguang Tong
Affliction: Australia National University, DATA61 CSIRO
Date: 2021-07-26 14:05:16
LastEditTime: 2021-07-27 15:44:47
'''

import torch
import os
import sys
sys.path.append(os.getcwd())
from torch import nn
from modules.functions import *

class Integrator(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.device = config.device

    def forward(self, feature: torch.Tensor, indices: torch.Tensor, feature_volume: torch.Tensor, count_volume: torch.Tensor):
        """[Integrate function]

        Integrate current feature v^t to global feature grid g^t
        
        Args:
            feature (torch.Tensor): local feature grid, N x H x W
            indices (torch.Tensor): indices position for each point in feature, N x H x W x 3
            feature_volume (torch.Tensor): global feature grid, X x Y x Z
            count_volume (torch.Tensor): update counts of global feature grid
        """
        xs, ys, zs = feature_volume.shape
        
        # reshape tensors
        n1, n2, n3 = feature.shape
        
        feature = feature.contiguous().view(n1 * n2 * n3, 1)
        weights = torch.ones_like(feature)
        indices = indices.contiguous().view(n1 * n2 * n3, 3).long()
        
        valid = get_index_mask(indices, feature_volume.shape)
        
        feature = torch.masked_select(feature[:, 0] ,valid)
        weights = torch.masked_select(weights[:, 0] ,valid)
        indices = extract_indices(indices, mask=valid)
        
        fcache = torch.zeros_like(feature_volume).view(xs * ys * zs).to(self.device)
        wcache = torch.zeros_like(feature_volume).view(xs * ys * zs).to(self.device)
        
        index = ys * zs * indices[:, 0] + zs * indices[:, 1] + indices[:, 2]
        
        fcache.index_add_(0, index, feature)
        wcache.index_add_(0, index, weights)
        
        fcache = fcache.view(xs, ys, zs)
        wcache = wcache.view(xs, ys, zs)
        
        update = extract_values(indices, fcache)
        weights = extract_values(indices, wcache)
        
        feature_pooling = update / weights
        
        feature_old = extract_values(indices, feature_volume)
        counts_old = extract_values(indices, count_volume)
        counts_update = counts_old + torch.ones_like(counts_old)
        
        feature_update = feature_old * counts_old + feature_pooling / counts_update
        
        insert_values(feature_update, indices, feature_volume)
        insert_values(counts_update, indices, count_volume)      
        
        return feature_volume, count_volume
        
if __name__ == '__main__':
    class Config(object):
        def __init__(self) -> None:
            self.device = torch.device('cpu')
    config = Config()
    integrator = Integrator(config)
    
    values = torch.rand(10, 128, 128)
    indices = (torch.rand(10, 128, 128, 3) * 120).long()
    weights = torch.randint(10, (10, 128, 128))
    values_volume = torch.rand(100, 100,100)
    count_volume = torch.randint(10, (100,100,100))
    res = integrator.forward(values, indices, values_volume, count_volume)
    print(res)