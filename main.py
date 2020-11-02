import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchcontrib.optim import SWA
from torch.nn.utils import rnn, clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import pdb
import json
import argparse
from argparse import Namespace
import numpy as np
from pesq import pesq
from tqdm import tqdm
from colorama import Fore
from collections import OrderedDict
from multiprocessing import Pool

from dataset import VoiceBankDemandDataset # please make your own "dataset.py" including a torch Dataset module.
from models import DeepConvolutionalUNet
from perceptual.losses import PerceptualLoss
from optimizers import RAdam


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cal_pesq(x, y, l):
    try:
        score = pesq(16000, y[:l], x[:l], 'wb')
    except:
        score = 0.
    return score

def evaluate(x, y, lens, fn):
    y = list(y.cpu().detach().numpy())
    x = list(x.cpu().detach().numpy())
    lens = lens.cpu().detach().tolist()
    pool = Pool(processes=args.num_workers)
    try:
        ret = pool.starmap(
            fn, 
            iter([(deg, ref, l) for deg, ref, l in zip(x, y, lens)])
        )
        pool.close()
        return torch.FloatTensor(ret).mean()

    except KeyboardInterrupt:
        pool.terminate()
        pool.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # system setting
    parser.add_argument('--exp_dir', default=os.getcwd(), type=str)
    parser.add_argument('--exp_name', default='logs', type=str)
    parser.add_argument('--data_dir', default='/Data/tahsieh/NSDTSEA/', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--add_graph', action='store_true')
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--seed', default=0, type=int)
    
    # training specifics
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--clip_grad_norm_val', default=0.0, type=float)
    parser.add_argument('--grad_accumulate_batches', default=1, type=int)
    parser.add_argument('--log_grad_norm', action='store_true')
    parser.add_argument('--resume_dir', default='', type=str)
    parser.add_argument('--use_swa', action='store_true')
    parser.add_argument('--lr_decay', default=1.0, type=float)

    # stft/istft settings
    parser.add_argument('--n_fft', default=512, type=int)
    parser.add_argument('--hop_length', default=128, type=int)
    
    # model hyperparameters
    parser.add_argument('--model_type', default='wav2vec', type=str)

    args = parser.parse_args()
    
    # add hyperparameters
    ckpt_path = os.path.join(args.exp_dir, args.exp_name, 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
        os.makedirs(ckpt_path.replace('ckpt', 'logs'))
        with open(os.path.join(ckpt_path, 'hparams.json'), 'w') as f:
            json.dump(vars(args), f)
    else:
        print(f'Experiment {args.exp_name} already exists.')
        sys.exit()
    writer = SummaryWriter(os.path.join(args.exp_dir, args.exp_name, 'logs'))
    writer.add_hparams(vars(args), dict())

    # seed
    if args.seed:
        fix_seed(args.seed)

    # device
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    if device == 'cuda':
        print(f'DEVICE: [{torch.cuda.current_device()}] {torch.cuda.get_device_name()}')
    else:
        print(f'DEVICE: CPU')

    # create loaders
    train_dataloader = DataLoader(
        # put your train dataset here,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataloader = DataLoader(
        # put your validation dataset here,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=rnn_collate,
        num_workers=args.num_workers
    )
    # create model
    if not args.resume_dir:
        net = DeepConvolutionalUNet(hidden_size=args.n_fft // 2 + 1)
        net = nn.DataParallel(net)
    else:
        try:
            with open(os.path.join(args.resume_dir, 'hparams.json'), 'r') as f:
                hparams = json.load(f)
        except FileNotFoundError:
            print('Cannot find "hparams.json".')
            sys.exit()

        hparams['resume_dir'] = args.resume_dir
        args = Namespace(**hparams)
        net = DeepConvolutionalUNet(hidden_size=args.n_fft // 2 + 1)
        net = nn.DataParallel(net)
        model_path = os.path.join(args.resume_dir, 'model_best.ckpt')
        print(f'Resume model from {model_path} ...')
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
    net = net.to(device)

    # optimization
    # optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=0.1)
    optimizer = RAdam(net.parameters(), lr=args.learning_rate, weight_decay=0.1)
    scheduler = None
    if args.use_swa:
        steps_per_epoch = len(train_dataloader) // args.batch_size
        optimizer = SWA(optimizer, swa_start=20 * steps_per_epoch, swa_freq=steps_per_epoch)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer.optimizer, mode="max", patience=5, factor=0.5)

    else:
        scheduler = None

    if args.resume_dir:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # best_pesq = checkpoint['pesq']
        best_pesq = 0.0
    else:
        start_epoch = 0
        best_pesq = 0.0
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    # add graph to tensorboard
    if args.add_graph:
        dummy = torch.randn(16, 1, args.hop_length * 16).to(device)
        writer.add_graph(net, dummy)

    # define loss
    cx_loss = PerceptualLoss(model_type=args.model_type)
    cx_loss = cx_loss.to(device)
    criterion = lambda y_hat, y: cx_loss(y_hat, y) + F.l1_loss(y_hat, y)

    # iteration start
    for epoch in range(start_epoch, start_epoch + args.num_epochs, 1):
        # ------------- training ------------- 
        net.train()
        pbar = tqdm(train_dataloader, bar_format='{l_bar}%s{bar}%s{r_bar}'%(Fore.BLUE, Fore.RESET))
        pbar.set_description(f'Epoch {epoch + 1}')
        total_loss = 0.0
        if args.log_grad_norm:
            total_norm = 0.0
        net.zero_grad()
        for i, (n, c) in enumerate(pbar):
            n, c = n.to(device), c.to(device)
            e = net(n)
            loss = criterion(e, c)
            loss /= args.grad_accumulate_batches
            loss.backward()

            # gradient clipping
            if args.clip_grad_norm_val > 0.0:
                clip_grad_norm_(net.parameters(), args.clip_grad_norm_val)

            # log metrics
            pbar_dict = OrderedDict({
                'loss': loss.item(),
            })
            pbar.set_postfix(pbar_dict)

            total_loss += loss.item()
            if (i + 1) % args.log_interval == 0:
                step = epoch * len(train_dataloader) + i
                writer.add_scalar('Loss/train', total_loss / args.log_interval, step)
                total_loss = 0.0

                # log gradient norm
                if args.log_grad_norm:
                    for p in net.parameters():
                        if p.requires_grad:
                            norm = p.grad.data.norm(2)
                            total_norm += norm.item() ** 2
                    norm = total_norm ** 0.5
                    writer.add_scalar('Gradient 2-Norm/train', norm, step)
                    total_norm = 0.0

            # accumulate gradients
            if (i + 1) % args.grad_accumulate_batches == 0:
                optimizer.step()
                net.zero_grad()

        # ------------- validation -------------
        pbar = tqdm(val_dataloader, bar_format='{l_bar}%s{bar}%s{r_bar}'%(Fore.LIGHTMAGENTA_EX, Fore.RESET))
        pbar.set_description('Validation')
        total_loss, total_pesq = 0.0, 0.0
        num_val_data = len(val_dataloader)
        with torch.no_grad():
            net.eval()
            for i, (n, c, l) in enumerate(pbar):
                n, c = n.to(device), c.to(device)
                e = net(n)
                loss = criterion(e, c)
                pesq_score = evaluate(e, c, l, fn=cal_pesq)
                pbar_dict = OrderedDict({
                    'val_loss': loss.item(),
                    'val_pesq': pesq_score.item(),
                })
                pbar.set_postfix(pbar_dict)

                total_loss += loss.item()
                total_pesq += pesq_score.item()

            if scheduler is not None:
                scheduler.step(total_pesq / num_val_data)

            writer.add_scalar('Loss/valid', total_loss / num_val_data, epoch)
            writer.add_scalar('PESQ/valid', total_pesq / num_val_data, epoch)

            # checkpointing
            curr_pesq = total_pesq / num_val_data
            if  curr_pesq > best_pesq:
                best_pesq = curr_pesq
                save_path = os.path.join(ckpt_path, 'model_best.ckpt')
                print(f'Saving checkpoint to {save_path}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss / num_val_data,
                    'pesq': total_pesq / num_val_data
                }, save_path)

    writer.flush()
    writer.close()
