'''
Description: 
Author: Jinguang Tong
Affliction: Australia National University, DATA61 CSIRO
Date: 2021-07-26 16:08:47
LastEditTime: 2021-07-27 19:40:07
'''

import torch


def get_index_mask(indices, shape):

    xs, ys, zs = shape

    valid = (
        (indices[:, 0] >= 0) &
        (indices[:, 0] < xs) &
        (indices[:, 1] >= 0) &
        (indices[:, 1] < ys) &
        (indices[:, 2] >= 0) &
        (indices[:, 2] < zs))

    return valid


def extract_indices(indices, mask):
    """
    extract indices w.r.t mask
    """
    mask = mask.unsqueeze(1)

    masked_indices = torch.masked_select(indices, mask).reshape(-1, 3)

    return masked_indices


def extract_values(indices, volume, mask=None, fusion_weights=None):

    if mask is not None:
        x = torch.masked_select(indices[:, 0], mask)
        y = torch.masked_select(indices[:, 1], mask)
        z = torch.masked_select(indices[:, 2], mask)
    else:
        x = indices[:, 0]
        y = indices[:, 1]
        z = indices[:, 2]

    return volume[x, y, z]


def insert_values(values, indices, volume):
    """

    insert values to volume according to indices

    Args:
        values ([type]): [description]
        indices ([type]): [description]
        volume ([type]): [description]
    """
    volume[indices[:, 0], indices[:, 1], indices[:, 2]] = values


def index_shift(radius):
    t = torch.arange(-radius, radius+1)
    dz, dy, dx = torch.meshgrid([t, t, t])
    shift = torch.stack([dx, dy, dz], dim=-1).view(-1, 3)
    return shift


def get_activation(module_name, class_name):
    import importlib
    m = importlib.import_module('torch.nn')
    return getattr(m, class_name)()


if __name__ == '__main__':
    print(index_shift(2))