'''
Description: 
Author: Jinguang Tong
Affliction: Australia National University, DATA61 CSIRO
Date: 2021-07-26 16:53:36
LastEditTime: 2021-07-26 17:03:26
'''

import yaml
import json
import os
import torch

from easydict import EasyDict

def load_config_from_yaml(path):
    """
    Method to load the config file for neural network training
    Args:
        path ([str]): [description]
    """
    c = yaml.load(open(path))
    return EasyDict(c)


if __name__ == '__main__':
    config = load_config_from_yaml("configs/fusion/shapenet.noise.005.yaml")
    print(config)
    