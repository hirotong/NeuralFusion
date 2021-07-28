'''
Description: 
Author: Jinguang Tong
Affliction: Australia National University, DATA61 CSIRO
Date: 2021-07-26 16:39:23
LastEditTime: 2021-07-27 22:23:05
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
        
        values = self._extractor.forward(frame, batch['extrinsics'], batch['intrinsics'], volume['current'], volume['origin'], volume['resolution'], volume['weights'])
        
        values_gt = self._extractor.forward(frame, batch['extrinsics'], batch['intrinsics'], volume['gt'], volume['origin'], volume['resolution'], volume['weights'])
        
        feature_input = self._prepare_fusion_input(frame, values, confidence)
        
        feature_target = values_gt['extracted_feature']
        feature_target = feature_target.view(b, h, w, -1)
        
        feature_est = self._fusion(feature_input, values)
            
        # reshape target
        
    
        # mask invalid losses
        feature_est = masking(feature_est, filtered_frame.view(b, h * w, 1))
         
        values['points'] = values['points'][:, :, :self.config.MODEL.n_points].contiguous()
        
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
        return feature_input    # b x 2 x h x w
    
    
    
    def _fusion(self, input, values):
        
        b, c, h, w = input.shape
        
        feature_pred = self._fusion_network.forward(input)
        feature_pred = feature_pred.permute(0, 2, 3, 1)
        
        output = dict()
        
        feature_est = feature_pred.view(b, h * w, -1)
        
        # TODO
        return feature_est
    
    
    def _prepare_volume_update(self, values, est, inputs):
        
        tail_points = self.config.MODEL.n_tail_points
        
        b, h, w = inputs.shape
        depth = inputs.view(b, h*w, 1)
        
        valid = (depth != 0.)
        valid = valid.nonzero()[:, 1]
        
        update_indices = values['indices'][:, valid, :tail_points, :, :]
        update_points = values['points'][:, valid, :tail_points, :]
        update_values = est[:, valid, :tail_points]
        
        update_values = torch.clamp(update_values, -self.config.DATA.init_value, self.config.DATA.init_value)
        
        del valid
        return update_values, update_indices, update_points
    
    

def masking(x, values, threshold=0., option='ueq'):
    
    if 'leq' == option:
        if x.dim() == 2:
            valid = (values <= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values <= threshold)