'''
Description: 
Author: Jinguang Tong
Affliction: Australia National University, CSIRO
Date: 2021-07-25 17:51:31
LastEditTime: 2021-07-31 16:59:48
'''

import torch
import torch.nn.functional as F
from torch import nn

import torchvision.models.resnet

def block(in_shape, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, activation=nn.Tanh()):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.LayerNorm(in_shape),
        activation,
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.LayerNorm(in_shape),
        activation,        
    )

class FusionNet(nn.Module):
    def __init__(self, config):
        super(FusionNet, self).__init__()
            
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
    def forward(self, x):
        x = x.float()
        x1 = self.encoder.forward(x)
        x2 = self.decoder.forward(x1) 
        # TODO how to normalize the feature
        x2 = F.normalize(x2, dim=1)    # b x (n_points * len_feature) x h x w
        return x2

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_shape = config.in_shape
        self.n_channels = config.n_points * config.len_feature + 4

        self.block1 = block(self.in_shape, self.n_channels, self.n_channels)
        self.block2 = block(self.in_shape, 2 * self.n_channels, self.n_channels)
        self.block3 = block(self.in_shape, 3 * self.n_channels, self.n_channels)
        self.block4 = block(self.in_shape, 4 * self.n_channels, self.n_channels)


    def forward(self, x):
        # every output tensor is concated with 
        # ? make sense?
        x1 = self.block1(x)
        x1 = torch.cat([x, x1], dim=1)
        x2 = self.block2(x1)
        x2 = torch.cat([x1, x2], dim=1)
        x3 = self.block3(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x4 = self.block4(x3)
        x4 = torch.cat([x3, x4], dim=1)
        
        del x1, x2, x3
        return x4
    

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_shape = config.in_shape 
        self.n_channels = config.n_points * config.len_feature + 4

        self.block1 = block(self.in_shape, 5 * self.n_channels, 4 * self.n_channels, kernel_size=1, padding=0)
        self.block2 = block(self.in_shape, 4 * self.n_channels, 3 * self.n_channels, kernel_size=1, padding=0)
        self.block3 = block(self.in_shape, 3 * self.n_channels, 2 * self.n_channels, kernel_size=1, padding=0)
        self.block4 = nn.Sequential(
        nn.Conv2d(2 * self.n_channels, self.n_channels,  kernel_size=1, padding=0, bias=True),
        nn.LayerNorm(self.in_shape),
        nn.Tanh(),
        nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, padding=0, bias=True),
        nn.LayerNorm(self.in_shape),
        nn.Tanh(),        
    )
        self.linear = nn.Conv2d(self.n_channels, config.n_points * config.len_feature, 1, 1)
        
    def forward(self, x):
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.linear(x)
        
        return x