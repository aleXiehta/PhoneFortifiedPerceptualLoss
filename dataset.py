import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import os
import sys


class VoiceBankDemandDataset(Dataset):
    def __init__(self, data_dir, hop_length=128):
        self.clean_path = self.get_clean(data_dir)
        self.noisy_path = self.get_noisy(data_dir)
        self.hop_length = hop_length

    def get_clean(self, root):
        raise NotImplementedError

    def get_noisy(self, root):
        raise NotImplementedError

    def padding(self, x):
        len_x = x.size(-1)
        pad_len = self.hop_length - len_x % self.hop_length
        x = F.pad(x, (0, pad_len))
        return x

    def normalize(self, x):
        return 2 * (x - x.min()) / (x.max() - x.min()) - 1

    def __len__(self):
        return len(self.noisy_path)

    def __getitem__(self, idx):
        clean = torchaudio.load(self.clean_path[idx])[0]
        noisy = torchaudio.load(self.noisy_path[idx])[0]

        noisy = self.normalize(noisy)
        length = clean.size(-1)
        clean.squeeze_(0)
        noisy.squeeze_(0)
        start = torch.randint(0, length - 16384 - 1, (1, ))
        end = start + 16384
        clean = clean[start:end]
        noisy = noisy[start:end]

        return noisy, clean
