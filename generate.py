import os
import sys
import json
import argparse

from argparse import Namespace
from colorama import Fore
from tqdm import tqdm

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

from dataset import VoiceBankDemandDataset
from models import DeepConvolutionalUNet
from utils import rnn_collate

if __name__ == '__main__':
    ckpt_root = sys.argv[1]
    save_root = sys.argv[2]
    hparams_path = os.path.join(ckpt_root, 'hparams.json')
    ckpt_path = os.path.join(ckpt_root, 'model_best.ckpt')

    if not os.path.exists(save_root):
        print(f'Making dir: {save_root}')
        os.makedirs(save_root)
    else:
        op = input('Do you want to overwrite this directory? [y/n]')
        if op == 'y':
            pass
        elif op == 'n':
            print(f'Directory {save_root} already exists. Process terminated')
            sys.exit()
        else:
            print('Invalid answer.')
            print(f'Directory {save_root} already exists. Process terminated')
            sys.exit()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f'DEVICE: [{torch.cuda.current_device()}] {torch.cuda.get_device_name()}')
    else:
        print(f'DEVICE: CPU')
    with open(os.path.join(hparams_path), 'r') as f:
        hparams = json.load(f)
    args = Namespace(**hparams)

    test_dataloader = DataLoader(
        # Put your test dataset here,
        batch_size=1,
        shuffle=False,
        collate_fn=rnn_collate,
        num_workers=args.num_workers
    )
    net = DeepConvolutionalUNet(hidden_size=args.n_fft // 2 + 1)
    net = nn.DataParallel(net)
    print(f'Resume model from {ckpt_path} ...')
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net = net.to(device)

    pbar = tqdm(test_dataloader, bar_format='{l_bar}%s{bar}%s{r_bar}'%(Fore.LIGHTMAGENTA_EX, Fore.RESET))
    pbar.set_description('Validation')
    total_loss, total_pesq = 0.0, 0.0
    num_test_data = len(test_dataloader)
    with torch.no_grad():
        net.eval()
        for i, (n, c, l) in enumerate(pbar):
            n, c = n.to(device), c.to(device)
            e = net(n)
            # e = (e + 1.) * (c.max() - c.min()) * 0.5 + c.min()
            assert e.sum() != 0
            assert e.max() <= 1.
            assert e.min() >= -1.
            filename = test_dataloader.dataset.noisy_path[i].split('/')[-1]
            pbar.set_postfix({'File name': filename})
            torchaudio.save(os.path.join(save_root, filename), e[:, :l].cpu(), 16000)
