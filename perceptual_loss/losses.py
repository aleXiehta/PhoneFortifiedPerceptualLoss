import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models.wav2vec import Wav2VecModel
import os
from functools import partial
from geomloss import SamplesLoss

class PerceptualLoss(nn.Module):
    def __init__(self, model_type='wav2vec', PRETRAINED_MODEL_PATH = '/path/to/wav2vec_large.pt'):
        super().__init__()
        self.model_type = model_type
        self.wass_dist = SamplesLoss()
        if model_type == 'wav2vec':
            ckpt = torch.load(PRETRAINED_MODEL_PATH)
            self.model = Wav2VecModel.build_model(ckpt['args'], task=None)
            self.model.load_state_dict(ckpt['model'])
            self.model = self.model.feature_extractor
            self.model.eval()
        else:
            print('Please assign a loss model')
            sys.exit()

    def forward(self, y_hat, y):
        y_hat, y = map(self.model, [y_hat, y])
        return self.wass_dist(y_hat, y)
        # for PFPL-W-MAE or PFPL-W
        # return torch.abs(y_hat - y).mean()
