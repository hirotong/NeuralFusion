'''
Description: 
Author: Jinguang Tong
Affliction: Australia National University, DATA61 CSIRO
Date: 2021-07-26 17:19:20
LastEditTime: 2021-08-03 18:25:13
'''

import torch
import torch.nn.functional as F
from modules.functions import *
from torch import nn


class Interpolator(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.count_threshold = config.count_threshold
        self.radius = config.radius
        self.n_neighbors = (2 * self.radius + 1) ** 3
        self.index_shift = index_shift(self.radius)
        self.linear = nn.Linear(
            self.n_neighbors * config.len_feature, config.len_feature)

    def _gather_feature(self, query_indices, query_points, feature_volume, count_volume, threshold=0):

        def _pad_feature_volume(feature_volume):
            feature_volume = feature_volume.permute(-1, 0, 1, 2)
            pad = tuple([self.radius] * 6)
            feature_volume = F.pad(feature_volume, pad)
            feature_volume = feature_volume.permute(1, 2, 3, 0)
            return feature_volume

        device = feature_volume.device
        xs, ys, zs, n = feature_volume.shape
        n1, n2, n3, _ = query_indices.shape
        
        self.index_shift = self.index_shift.to(device)
        query_indices = query_indices.contiguous().view(n1 * n2 * n3, 3).long()
        neighbor_indices = (query_indices.unsqueeze(1) + self.index_shift).contiguous().view(-1, 3)
        
        # ? whether to filter via update count
        # valid = torch.gt(count_volume, threshold)
        # indices = torch.nonzero(valid)

        # shift radius due to feature padding below
        neighbor_indices = neighbor_indices + self.radius

        # pad feature volume in case of out of bound
        padded_feature = _pad_feature_volume(feature_volume)

        gathered_feature = extract_values(neighbor_indices, padded_feature)
        gathered_feature = gathered_feature.view(-1, self.n_neighbors * n)

        del padded_feature

        return gathered_feature, query_indices

    def forward(self, query_indices, query_points, feature_volume, count_volume):

        # xs, ys, zs, n = feature_volume.shape

        gathered_feature, indices = self._gather_feature(query_indices, query_points, feature_volume, count_volume, self.count_threshold)
        gathered_feature = gathered_feature.float()

        gathered_feature = self.linear(gathered_feature)

        query_feature = extract_values(indices, feature_volume)
        query_feature = query_feature.float()

        return torch.cat([query_feature, gathered_feature], dim=-1), indices

class TranslateMLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.tsdf_scale = config.tsdf_scale
        self.p = config.p_dropout
        self.n_points = config.n_points
        self.len_feature = config.len_feature
        self.activation = get_activation('torch.nn', config.activation)
        self.out_channels = [32, 16, 8, 8]
        self.layer1 = nn.Sequential(
            nn.Linear(self.len_feature * 2, self.out_channels[0]),
            self.activation,
            nn.Dropout(p=self.p)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(self.out_channels[0] + self.len_feature * 2, self.out_channels[1]),
            self.activation,
            nn.Dropout(p=self.p)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(self.out_channels[1] + self.len_feature * 2, self.out_channels[2]),
            self.activation,
            nn.Dropout(p=self.p)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(self.out_channels[2] + self.len_feature * 2, self.out_channels[3]),
            self.activation,
            nn.Dropout(p=self.p)
        )

        # TODO The output of the activation is further scaled by the truncation band of the ground-truth TSDF (0.04) to map it into the correct value range.         
        self.tsdf_head = nn.Sequential(
            nn.Linear(self.out_channels[3], 1),
            nn.Tanh()
        )

        self.occ_head = nn.Sequential(
            nn.Linear(self.out_channels[3], 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x1 = self.layer1(x)
        x1 = torch.cat([x, x1], dim=1)
        x2 = self.layer2(x1)
        x2 = torch.cat([x, x2], dim=1)
        x3 = self.layer3(x2)
        x3 = torch.cat([x, x3], dim=1)
        x4 = self.layer4(x3)
        
        tsdf =self.tsdf_scale * self.tsdf_head(x4)
        occupancy = self.occ_head(x4)

        return tsdf, occupancy
        


class Translator(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self._neighbor_interpolator = Interpolator(config)
        self._translate_mlp = TranslateMLP(config)
        
    def forward(self, query_indices, query_points, feature_volume, count_volume):
        
        device = feature_volume.device
        # tsdf_volume = torch.zeros_like(count_volume).float().to(device)
        # occ_volume = torch.zeros_like(count_volume).float().to(device)
        gathered_feature, indices = self._neighbor_interpolator.forward(query_indices, query_points,feature_volume, count_volume)
        tsdf, occupancy = self._translate_mlp.forward(gathered_feature)

        # insert_values(tsdf.squeeze(), indices, tsdf_volume)
        # insert_values(occupancy.squeeze(), indices, occ_volume)
        del gathered_feature, indices
        return tsdf, occupancy # tsdf_volume, occ_volume, 
    