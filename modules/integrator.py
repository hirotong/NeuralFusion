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
            feature (torch.Tensor): local feature grid, N x n x H x W
            indices (torch.Tensor): indices position for each point in feature, N x H x W x 3
            feature_volume (torch.Tensor): global feature grid, X x Y x Z
            count_volume (torch.Tensor): update counts of global feature grid
        """
        xs, ys, zs, n = feature_volume.shape
        
        # reshape tensors
        n1, n2, n3 = feature.shape
        
        feature = feature.contiguous().view(-1, n)
        weights = torch.ones_like(feature)
        indices = indices.contiguous().view(n1 * n2 * n, 3).long()
        
        valid = get_index_mask(indices, feature_volume.shape)
        
        feature = torch.masked_select(feature ,valid.unsqueeze(1)).view(-1, n)
        weights = torch.masked_select(weights ,valid.unsqueeze(1)).view(-1, n)
        indices = extract_indices(indices, mask=valid)
        
        fcache = torch.zeros_like(feature_volume, dtype=torch.float).view(xs * ys * zs, n).to(self.device)
        wcache = torch.zeros_like(feature_volume, dtype=torch.float).view(xs * ys * zs, n).float().to(self.device)
        
        index = ys * zs * indices[:, 0] + zs * indices[:, 1] + indices[:, 2]
        
        fcache.index_add_(0, index, feature)
        wcache.index_add_(0, index, weights)
        
        fcache = fcache.view(xs, ys, zs, n)
        wcache = wcache.view(xs, ys, zs, n)
        
        update = extract_values(indices, fcache)
        weights = extract_values(indices, wcache)
        
        feature_pooling = update / weights
        
        feature_old = extract_values(indices, feature_volume)
        counts_old = extract_values(indices, count_volume)
        counts_update = counts_old + torch.ones_like(counts_old)
        
        feature_update = (feature_old * counts_old.unsqueeze(1) + feature_pooling) / counts_update.unsqueeze(1)
        
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