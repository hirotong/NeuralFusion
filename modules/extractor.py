import torch
import datetime
import numpy as np
from torch import nn
from modules.functions import *
import torch.nn.functional as F


# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Extractor(nn.Module):
    """

    """

    def __init__(self, config):
        super(Extractor, self).__init__()
        
        self.config = config
        
        self.n_points = config.n_points
        self.mode = 'ray'

    def forward(self, depth, extrinsics, intrinsics, feature_volume, origin, resolution, weight_volume=None):
        """

        Compute the forward pass of extracting the rays/blocks and the corresponding coordinates

        Args:
            depth ([type]): [description]
            extrinsics ([type]): [description]
            intrinsics ([type]): [description]
            feature_volume ([type]): [description]
            origin ([type]): [description]
            resolution ([type]): [description]
            weight_volume ([type], optional): [description]. Defaults to None.
        """

        device = depth.get_device()
        
        feature_volume = feature_volume.float()
        intrinsics = intrinsics.float()
        extrinsics = extrinsics.float()

        if device >= 0:
            intrinsics = intrinsics.to(device)
            extrinsics = extrinsics.to(device)

            feature_volume = feature_volume.to(device)
            # weight_volume = weight_volume.to(device)
            origin = origin.to(device)

        b, h, w = depth.shape

        self.depth = depth.contiguous().view(b, h * w)

        # world coordinates
        coords = self.compute_coordinates(depth, extrinsics, intrinsics, origin, resolution)    # b x (h * w) x 3
        
        # compute rays
        # camera in world coordinate
        eye_w = extrinsics[:, :3, 3]

        ray_pts, ray_dists, ray_directions = self._extract_values(coords, eye_w, origin, resolution, n_points=int((self.n_points-1)/2))
        
        extracted_feature, indices = sample_feature(ray_pts, feature_volume, 'nearest')

        extracted_feature = extracted_feature.squeeze(-1)
        n1, n2, n3 = extracted_feature.shape[:3]
        
        indices = indices.view(n1, n2, n3, 3)
        
        # packing
        values = dict(extracted_feature=extracted_feature,
                      points=ray_pts,
                      direction=ray_directions,
                      depth=depth.view(b, h*w),
                      indices=indices,
                      pcl=coords)
        
        del extrinsics, intrinsics, origin, weight_volume, feature_volume
        
        return values

 
    def compute_coordinates(self, depth, extrinsics, intrinsics, origin, resolution):
        """
        Un-project points in the depth image (p_x, p_y, d)^T to world coordinates

        :param depth: depth images
        :param extrinsics: inverted camera extrinsics matrix for mapping (P_c -> P_w)
        :param intrinsics:
        :param origin:
        :param resolution:
        :return: points_w (b x n_points x 3)
        """

        device = depth.device
        b, h, w = depth.shape
        n_points = h * w

        # generate frame meshgrid
        xx, yy = torch.meshgrid([torch.arange(h, dtype=torch.float),
                                 torch.arange(w, dtype=torch.float)])

        xx = xx.to(device)
        yy = yy.to(device)

        # flatten grid coordinates and bring them to batch size
        xx = xx.contiguous().view(1, n_points, 1).repeat((b, 1, 1))
        yy = yy.contiguous().view(1, n_points, 1).repeat((b, 1, 1))
        zz = depth.contiguous().view(b, n_points, 1)

        # generate points in pixel space
        points_p = torch.cat((yy, xx, zz), dim=2).clone()

        # invert
        intrinsics_inv = intrinsics.inverse().float()

        homogeneous = torch.ones((b, 1, n_points)).to(device)

        # transform points from pixel space to camera space to world space (p -> c -> w)
        # firstly, xx & yy coordinate should multiple zz
        points_p[:, :, 0] *= zz[:, :, 0]
        points_p[:, :, 1] *= zz[:, :, 0]
        # camera space
        points_c = torch.matmul(intrinsics_inv,
                                torch.transpose(points_p, dim0=1, dim1=2))
        points_c = torch.cat((points_c, homogeneous), dim=1)
        # world space
        points_w = torch.matmul(extrinsics[:, :3], points_c)
        points_w = torch.transpose(points_w, dim0=1, dim1=2)[:, :, :3]

        del xx, yy, homogeneous, points_p, points_c, intrinsics_inv
        return points_w

    def _extract_values(self, coords, eye, origin, resolution, bin_size=1.0,
                       n_points=4):
        """

        :param coords:
        :param eye:
        :param origin:
        :param resolution: 1 / grid_resolution
        :param bin_size:
        :param n_points:
        :return: points: extracted points coords # b x h*w x 2*n_points+1 x 3
        """
        center_v = (coords - origin) / resolution
        eye_v = (eye - origin) / resolution

        direction = center_v - eye_v.unsqueeze(1)
        direction = F.normalize(direction, p=2, dim=2)

        points = [center_v]

        dist = torch.zeros_like(center_v)[:, :, 0]
        dists = [dist]

        # sample points along the ray direction
        for i in range(1, n_points + 1):
            pointP = center_v + i * bin_size * direction
            pointN = center_v - i * bin_size * direction
            points.append(pointP.clone())
            points.insert(0, pointN.clone())

            distP = i * bin_size * torch.ones_like(pointP)[:, :, 0]
            distN = -1 * distP

            dists.append(distP)
            dists.insert(0, distN)

        dists = torch.stack(dists, dim=2)
        points = torch.stack(points, dim=2)

        return points, dists, direction


def sample_feature(points, feature_volume, method='nearset'):
    if 'nearest' == method:
        return nearest_feature(points, feature_volume)
    elif 'trilinear' == method:
        return trilinear_feature(points, feature_volume)
    else:
        raise NotImplementedError(f"{method} is not supported.")


def nearest_feature(points, feature_volume):

    device = feature_volume.device
    b, h, n, dim = points.shape

    # get neareast indices

    indices = nearest_indices(points)   # b x (h*n) x dim

    n1, n2, n3 = indices.shape
    indices = indices.contiguous().view(n1 * n2, n3)

    # get valid indices
    valid = get_index_mask(indices, feature_volume.shape)
    valid_idx = torch.nonzero(valid)[:, 0]

    feature_values = extract_values(indices, feature_volume, valid)
    feature_values = feature_values.view(feature_values.shape[0], -1)
    feature_container = torch.zeros((valid.shape[0], feature_values.shape[1]), dtype=torch.float, device=device)
    feature_container[valid_idx] = feature_values

    feature_values = feature_container.view(b, h, n, -1)
    
    del feature_container
    
    return feature_values.float(), indices


def trilinear_feature(points, feature_volume, weight_volume=None):

    b, h, w, dim = points.shape

    # get interpolation weights
    weights, indices = interpolation_weights(points)

    n1, n2, n3 = indices.shape
    indices = indices.contiguous().view(n1 * n2, n3).long()

    # get valid indices
    valid = get_index_mask(indices, feature_volume.shape)
    valid_idx = torch.nonzero(valid)[:, 0]

    tsdf_values = extract_values(indices, feature_volume, valid)
    tsdf_weights = extract_values(indices, weight_volume, valid)

    value_container = torch.zeros_like(valid).double()
    weight_container = torch.zeros_like(valid).double()

    value_container[valid_idx] = tsdf_values
    weight_container[valid_idx] = tsdf_weights

    value_container = value_container.view(weights.shape)
    weight_container = weight_container.view(weights.shape)

    # trilinear interpolation
    fusion_values = torch.sum(value_container * weights, dim=1)
    fusion_weights = torch.sum(weight_container * weights, dim=1)

    fusion_values = fusion_values.view(b, h, w)
    fusion_weights = fusion_weights.view(b, h, w)

    indices = indices.view(n1, n2, n3)

    return fusion_values.float(), indices, weights, fusion_weights.float()


def interpolation_weights(points, mode='center'):
    if 'center' == mode:
        # compute step direction
        center = torch.floor(points) + 0.5 * torch.ones_like(points)
        # this is different from the author's implementation
        neighbor = torch.sign(points - center)
    else:
        center = torch.floor(points)
        neighbor = torch.sign(points - center)

    # index of center voxel
    idx = torch.floor(points)

    # reshape for pytorch compatibility
    b, h, w, dim = idx.shape
    points = points.contiguous().view(b * h * w, dim)
    center = center.contiguous().view(b * h * w, dim)
    idx = idx.view(b * h * w, dim)
    neighbor = neighbor.view(b * h * w, dim)

    # center x.0
    alpha = torch.abs(points - center)  # always positive
    alpha_inv = 1 - alpha

    weights = []
    indices = []

    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                if i == 0:
                    w1 = alpha_inv[:, 0]
                    ix = idx[:, 0]
                else:
                    w1 = alpha[:, 0]
                    ix = idx[:, 0] + neighbor[:, 0]
                if j == 0:
                    w2 = alpha_inv[:, 1]
                    iy = idx[:, 1]
                else:
                    w2 = alpha[:, 1]
                    iy = idx[:, 1] + neighbor[:, 1]
                if k == 0:
                    w3 = alpha_inv[:, 2]
                    iz = idx[:, 2]
                else:
                    w3 = alpha[:, 2]
                    iz = idx[:, 2]
                weights.append((w1 * w2 * w3).unsqueeze_(1))
                indices.append(torch.cat((ix.unsqueeze_(1),
                                          iy.unsqueeze_(1),
                                          iz.unsqueeze_(1)), dim=1).unsqueeze_(1))
    weights = torch.cat(weights, dim=1)
    indices = torch.cat(indices, dim=1)

    del points, center, idx, neighbor, alpha, alpha_inv, ix, iy, iz, w1, w2, w3

    return weights, indices


def nearest_indices(points, mode='center'):
    
    b, h, n, dim = points.shape
    
    if 'center' == mode:
        center = torch.floor(points) + 0.5 * torch.ones_like(points)
        indices = torch.floor(points)
    else:
        center = torch.floor(points)
        neighbor = torch.gt(points - center, 0.5 *
                            torch.ones_like(points)).float()
        indices = torch.floor(points) + neighbor
    indices = indices.contiguous().view(b, h*n, dim).long()
    
    return indices


if __name__ == '__main__':
    ex = Extractor('a')
    depth = torch.rand(10, 128, 128)
    extrinsics = torch.rand(10, 4, 4)
    intrinsics = torch.eye(3)

    tsdf_volume = torch.rand(500, 500, 500)
    origin = torch.Tensor([-1., -1., -1.])
    result = ex.forward(depth, extrinsics, intrinsics, tsdf_volume, origin, 1 / 128 )

    points = torch.rand(2, 10, 10, 3) * 128
    weights_volume = torch.rand(100, 100, 100).double()
    print(result)
    # trilinear_feature(points, tsdf_volume, weights_volume)
