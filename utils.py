import torch
from torch.nn.utils import rnn


def rnn_collate(batch):
        n = rnn.pad_sequence([b[0] for b in batch]).transpose(0, 1)
        c = rnn.pad_sequence([b[1] for b in batch]).transpose(0, 1)
        l = torch.LongTensor([b[2] for b in batch])
        return n, c, l
