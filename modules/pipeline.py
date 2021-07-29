'''
Author: Jinguang Tong
Affliction: Australia National University, DATA61 CSIRO
'''

import torch

from torch import nn
from modules.extractor import Extractor
from modules.fusion import FusionNet
from modules.integrator import Integrator
from modules.translator import Translator


class Pipeline(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self._extractor = Extractor(config.MODEL)
        self._fusion_network = FusionNet(config.MODEL)
        self._integrator = Integrator(config.MODEL)
        self._translator = Translator(config.MODEL)

    def fuse_training(self, batch, database, device):
        output = dict()
        self.device = device

        frame = batch[self.config.DATA.input].squeeze_(1)
        frame = frame.to(device)
        confidence = None

        mask = batch['mask'].to(device)
        filtered_frame = torch.where(mask == 0, torch.zeros_like(frame), frame)

        b, h, w = frame.shape

        # get current feature values
        scene_id = batch['scene_id'][0]
        volume = database[scene_id]

        values = self._extractor.forward(frame, batch['extrinsics'], batch['intrinsics'],
                                         volume['current'], volume['origin'], volume['resolution'], volume['weights'])

        tsdf_gt = self._extractor.forward(frame, batch['extrinsics'], batch['intrinsics'], volume['tsdf'],
                                            volume['origin'], volume['resolution'], volume['weights'])
        occ_gt = self._extractor.forward(frame,  batch['extrinsics'], batch['intrinsics'], volume['occ'],
                                            volume['origin'], volume['resolution'], volume['weights'])

        feature_input = self._prepare_fusion_input(frame, values, confidence)


        feature_est = self._fusion(feature_input, values)
        
        # reshape target
        tsdf_target = tsdf_gt['extracted_feature']
        occ_target = occ_gt['extracted_feature']
        tsdf_target = tsdf_target.view(b, h, w, -1)
        occ_target = occ_target.view(b, h, w, -1)
        # feature_target = feature_target.view(b, h * w, self.config.MODEL.n_points)
        
        values['points'] = values['points'][:, :, :self.config.MODEL.n_points].contiguous()
        # mask invalid losses
        feature_est = masking(feature_est, filtered_frame.view(b, h * w, 1))

        integration_values, integration_indices, integration_points = self._prepare_integration_input(
            values, feature_est, filtered_frame)

        feature_volume, count_volume = self._integrator.forward(
            integration_values.to(device),
            integration_indices.to(device),
            database[batch['scene_id'][0]]['current'].to(device),
            database[batch['scene_id'][0]]['counts'].to(device))
        database.secens_est[batch['scene_id'][0]
                            ].volume = feature_volume.cpu().detach().numpy()
        database.update_counts[batch['scene_id'][0]
                               ] = count_volume.cpu().detach().numpy()

        # translate
        
        tsdf_volume, occ_volume, tsdf_est, occ_est = self._translator.forward(integration_indices, integration_points, feature_volume, count_volume)
        
        output['tsdf_est'] = tsdf_est
        output['occ_est'] = occ_est
        output['tsdf_target'] = tsdf_target
        output['occ_target'] = occ_target
        
        return output


        # TODO

    def _prepare_fusion_input(self, frame, values, confidence=None):
        # get frame shape
        b, h, w = frame.shape

        # extracting data
        feature_input = values['extracted_feature']

        # reshaping data
        feature_input = feature_input.view(b, h, w, -1)

        feature_frame = torch.unsqueeze(frame, -1)

        # stacking input data
        feature_input = torch.cat([feature_input, feature_frame], dim=3)

        # permuting input
        feature_input = feature_input.permute(0, 3, 1, 2)

        del feature_frame
        return feature_input  # b x 2 x h x w

    def _fusion(self, input, values):
        b, c, h, w = input.shape

        feature_pred = self._fusion_network.forward(input)
        feature_pred = feature_pred.permute(0, 2, 3, 1)

        output = dict()

        feature_est = feature_pred.view(b, h * w, -1)

        # TODO
        return feature_est

    def _prepare_integration_input(self, values, est, inputs):
        tail_points = self.config.MODEL.n_tail_points

        b, h, w = inputs.shape
        depth = inputs.view(b, h * w, 1)

        valid = (depth != 0.)
        valid_idx = valid.nonzero()[:, 1]

        integration_indices = values['indices'][:, valid]
        integration_points = values['points'][:, valid]
        integration_values = est[:, valid]

        # TODO clamp the integration_values
        del valid

        return integration_values, integration_indices, integration_points

    def _prepare_translation_input(self, ):
        pass


def masking(x, values, threshold=0., option='ueq'):
    if 'leq' == option:
        if x.dim() == 2:
            valid = (values <= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values <= threshold)
