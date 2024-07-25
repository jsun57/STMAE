import os
import sys
import csv
import math
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from einops import rearrange
import matplotlib.pyplot as plt

sys.path.append('../')
from data_utils.dataset_pems import load_dataset
from utils.log_utils import save_csv_log
from utils.torch_utils import get_scheduler
from utils.loss_utils import masked_mae, masked_mape, masked_rmse, metric

def exists(x):
    return x is not None


class FineTuner(object):
    """
    Follow STGCL s_train.py for now
    """
    def __init__(
        self,
        dataset,
        pretrained_encoder,
        decoder,
        adj,
        loss,
        cfg,
        batch_size,
        learning_rate_enc,
        learning_rate_dec,
        weight_decay_enc,
        weight_decay_dec,
        ft=True,                    # default, finetuner finetune the model; if not, only train the decoder
        opt_name='adam',
    ):
        super().__init__()

        self.pretrained_encoder = pretrained_encoder
        self.decoder = decoder
        self.device = next(self.pretrained_encoder.parameters()).device
        self.adj = adj              # The adjacency matrix is from https://arxiv.org/pdf/2108.11873.pdf: weighted with self_loops
        self.loss = loss

        self.cfg = cfg

        self.ft = ft
        self.batch_size = batch_size
        self.opt_name = opt_name
        self.learning_rate_enc = learning_rate_enc
        self.learning_rate_dec = learning_rate_dec
        self.weight_decay_enc = weight_decay_enc
        self.weight_decay_dec = weight_decay_dec

        self.dataset_name = cfg.dataset
        self.in_horizon = cfg.in_horizon
        self.out_horizon = cfg.out_horizon
        self.dtype = torch.float32 if self.cfg.dtype == 'float32' else torch.float64

        self.dataset_name = cfg.dataset

        # datasets
        if self.dataset_name == 'pems_03':
            input_data = './data/pems_03'
        elif self.dataset_name == 'pems_04':
            input_data = './data/pems_04'
        elif self.dataset_name == 'pems_07':
            input_data = './data/pems_07'
        elif self.dataset_name == 'pems_08':
            input_data = './data/pems_08'
        else:
            raise NotImplementedError

        self.dataloader = load_dataset(input_data, self.batch_size, self.batch_size, self.batch_size, device=self.device, dtype=self.dtype)
        self.scaler = self.dataloader['scaler'] 

        # optimizer
        optimizer = Adam if self.opt_name == 'adam' else AdamW
        if self.ft:
            self.opt = optimizer(
                [{'params': self.pretrained_encoder.parameters(), 'lr': self.learning_rate_enc, 'weight_decay': self.weight_decay_enc, 'eps': self.cfg.eps_enc},
                {'params': self.decoder.parameters(), 'lr': self.learning_rate_dec, 'weight_decay': self.weight_decay_dec, 'eps': self.cfg.eps_dec}]
            )
        else:
            self.opt = optimizer(self.decoder.parameters(), lr=self.learning_rate_dec, weight_decay=self.weight_decay_dec, eps=self.cfg.eps_dec)
        self.scheduler = get_scheduler(self.opt, policy=self.cfg.ft_sched_policy, nepoch_fix=self.cfg.ft_num_epoch_fix_lr, nepoch=self.cfg.ft_epoch, \
        decay_step=self.cfg.ft_decay_step, gamma=self.cfg.ft_gamma, milestones=self.cfg.ft_milestones)

        self.epoch = 0
        self.train_loss_list = []
        self.valid_loss_list = []
        self.best_valid_loss = float('inf')
        self.batches_seen = None
        print('Finetuner initialization done.')

    def _get_error_info(self, prediction, target):
        mae = masked_mae(prediction, target, 0.0).item()
        rmse = masked_rmse(prediction, target, 0.0).item()
        mape = masked_mape(prediction, target, 0.0).item()
        error_info = {'mae': mae, 'rmse': rmse, 'mape': mape}
        return error_info

    def save(self, to_save_path):
        data = {
            'epoch': self.epoch,
            'train_loss_list': self.train_loss_list,
            'valid_loss_list': self.train_loss_list,
            'best_valid_loss': self.best_valid_loss,
            'batches_seen': self.batches_seen,
            'pretrained_encoder': self.pretrained_encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'opt': self.opt.state_dict(),
            'sched': self.scheduler.state_dict() if exists(self.scheduler) else None,
        }
        torch.save(data, to_save_path)
        return

    def load(self, to_load_path):
        if self.cfg.backbone == 'dcrnn':
            self.pretrained_encoder(torch.rand(1, self.in_horizon, self.cfg.num_nodes, self.cfg.in_dim).to(self.device).to(self.dtype))
            self.decoder(torch.rand(1, self.in_horizon, self.cfg.num_nodes, self.cfg.rnn_dim).to(self.device).to(self.dtype), 
            torch.rand(1, self.out_horizon, self.cfg.num_nodes, self.cfg.out_dim).to(self.device).to(self.dtype), batches_seen=0)
            optimizer = Adam if self.opt_name == 'adam' else AdamW
            if self.ft:
                self.opt = optimizer(
                    [{'params': self.pretrained_encoder.parameters(), 'lr': self.learning_rate_enc, 'weight_decay': self.weight_decay_enc, 'eps': self.cfg.eps_enc},
                    {'params': self.decoder.parameters(), 'lr': self.learning_rate_dec, 'weight_decay': self.weight_decay_dec, 'eps': self.cfg.eps_dec}]
                )
            else:
                self.opt = optimizer(self.decoder.parameters(), lr=self.learning_rate_dec, weight_decay=self.weight_decay_dec, eps=self.cfg.eps_dec)
            self.scheduler = get_scheduler(self.opt, policy=self.cfg.ft_sched_policy, nepoch_fix=self.cfg.ft_num_epoch_fix_lr, nepoch=self.cfg.ft_epoch, \
            decay_step=self.cfg.ft_decay_step, gamma=self.cfg.ft_gamma, milestones=self.cfg.ft_milestones)
        device = self.device
        data = torch.load(to_load_path, map_location=device)
        self.epoch = data['epoch']
        self.train_loss_list = data['train_loss_list']
        self.valid_loss_list = data['valid_loss_list']
        self.best_valid_loss = data['best_valid_loss']
        self.batches_seen = data['batches_seen']
        self.pretrained_encoder.load_state_dict(data['pretrained_encoder'])
        self.decoder.load_state_dict(data['decoder'])
        self.opt.load_state_dict(data['opt'])
        if exists(data['sched']):
            self.scheduler.load_state_dict(data['sched'])  
        else: self.scheduler = None
        print(">>> finish loading pretrained-encoder, decoder model ckpt from path '{}'".format(to_load_path))
        return


    def test_load(self, to_load_path):
        if self.cfg.backbone == 'dcrnn':
            self.pretrained_encoder(torch.rand(1, self.in_horizon, self.cfg.num_nodes, self.cfg.in_dim).to(self.device).to(self.dtype))
            self.decoder(torch.rand(1, self.in_horizon, self.cfg.num_nodes, self.cfg.rnn_dim).to(self.device).to(self.dtype), 
            torch.rand(1, self.out_horizon, self.cfg.num_nodes, self.cfg.out_dim).to(self.device).to(self.dtype), batches_seen=0)
        device = self.device
        data = torch.load(to_load_path, map_location=device)
        self.pretrained_encoder.load_state_dict(data['pretrained_encoder'])
        self.decoder.load_state_dict(data['decoder'])
        print(">>> finish loading pretrained-encoder, decoder model ckpt from path '{}'".format(to_load_path))
        return

    def train(self):
        """
        finetune model for one epoch
        """ 
        if self.ft:
            self.pretrained_encoder.train()
        else:
            self.pretrained_encoder.eval()
        self.decoder.train()
        t_s = time.time()
        epoch_loss = 0.
        epoch_iter = 0
        epoch_error_info = {}
        self.batches_seen = self.dataloader['train_loader'].num_batch * self.epoch    # this is for dcrnn specific training
        self.dataloader['train_loader'].shuffle()
        
        for idx, (x, y) in enumerate(self.dataloader['train_loader'].get_iterator()):
            x = x[...,:self.cfg.in_dim]
            y = y[...,:self.cfg.out_dim]
            all_states, encoded_state, _, _, _ = self.pretrained_encoder.encode(x, mask_s=0, mask_f=0)
            if self.cfg.backbone == 'dcrnn':
                output = self.decoder(all_states, self.scaler.transform(y), self.batches_seen)
            elif self.cfg.backbone == 'gwnet' or self.cfg.backbone == 'mtgnn':
                output = self.decoder(encoded_state)
            else:
                output = self.decoder(encoded_state.unsqueeze(1))
            output = self.scaler.inverse_transform(output)
            loss = self.loss(output, y[...,:self.cfg.out_dim], 0.0)
            error_info = self._get_error_info(output, y[...,:self.cfg.out_dim])
            for key, value in error_info.items():
                if key not in epoch_error_info:
                    epoch_error_info[key] = value
                else:
                    epoch_error_info[key] += value

            if self.batches_seen == 0 and self.cfg.backbone == 'dcrnn':   # dcrnn only
                optimizer = Adam if self.opt_name == 'adam' else AdamW
                if self.ft:
                    self.opt = optimizer(
                        [{'params': self.pretrained_encoder.parameters(), 'lr': self.learning_rate_enc, 'weight_decay': self.weight_decay_enc, 'eps': self.cfg.eps_enc},
                        {'params': self.decoder.parameters(), 'lr': self.learning_rate_dec, 'weight_decay': self.weight_decay_dec, 'eps': self.cfg.eps_dec}]
                    )
                else:
                    self.opt = optimizer(self.decoder.parameters(), lr=self.learning_rate_dec, weight_decay=self.weight_decay_dec, eps=self.cfg.eps_dec)
                self.scheduler = get_scheduler(self.opt, policy=self.cfg.ft_sched_policy, nepoch_fix=self.cfg.ft_num_epoch_fix_lr, nepoch=self.cfg.ft_epoch, \
                decay_step=self.cfg.ft_decay_step, gamma=self.cfg.ft_gamma, milestones=self.cfg.ft_milestones)

            self.opt.zero_grad()
            loss.backward()
            if exists(self.cfg.ft_clip_grad) and self.cfg.ft_clip_grad != 'None':
                nn.utils.clip_grad_norm_(self.pretrained_encoder.parameters(), max_norm=self.cfg.ft_clip_grad)
                nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.cfg.ft_clip_grad)
            self.opt.step()
            epoch_loss += loss.item()
            self.batches_seen += 1
            epoch_iter += 1
        if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and exists(self.scheduler):
            self.scheduler.step()
        
        self.epoch += 1
        epoch_loss /= epoch_iter
        for key, value in epoch_error_info.items():
            epoch_error_info[key] /= epoch_iter
        if self.ft:
            en_lr = self.opt.param_groups[0]['lr']
            de_lr = self.opt.param_groups[1]['lr']
        else:
            en_lr = -1
            de_lr = self.opt.param_groups[0]['lr']
        dt = time.time() - t_s
        self.train_loss_list.append(epoch_loss)
        return en_lr, de_lr, epoch_loss, epoch_error_info, dt
    
    @torch.no_grad()
    def valid(self):
        self.pretrained_encoder.eval()
        self.decoder.eval()
        total_iter = 0
        epoch_error_info = {}
        for idx, (x, y) in enumerate(self.dataloader['val_loader'].get_iterator()):
            x = x[...,:self.cfg.in_dim]
            y = y[...,:self.cfg.out_dim]
            all_states, encoded_state, _, _, _ = self.pretrained_encoder.encode(x, mask_s=0, mask_f=0)
            if self.cfg.backbone == 'dcrnn':
                output = self.decoder(all_states)
            elif self.cfg.backbone == 'gwnet' or self.cfg.backbone == 'mtgnn':
                output = self.decoder(encoded_state)
            else:
                output = self.decoder(encoded_state.unsqueeze(1))
            output = self.scaler.inverse_transform(output)
            error_info = self._get_error_info(output, y[...,:self.cfg.out_dim])
            for key, value in error_info.items():
                if key not in epoch_error_info:
                    epoch_error_info[key] = value
                else:
                    epoch_error_info[key] += value
            total_iter += 1

        for key, value in epoch_error_info.items():
            epoch_error_info[key] /= total_iter
        self.valid_loss_list.append(epoch_error_info['mae'])
        return epoch_error_info

    @torch.no_grad()
    def test(self):
        """
        NOTE: test should only be called once
        """
        # set header and logger
        head = np.array(['metric'])
        for k in range(1, self.out_horizon + 1):
            head = np.append(head, [f'{k}'])
        log = np.zeros([4, self.out_horizon + 1])

        self.pretrained_encoder.eval()
        self.decoder.eval()
        total_iter = 0
        all_preds, all_targets = [], []
        for idx, (x, y) in enumerate(self.dataloader['test_loader'].get_iterator()):
            x = x[...,:self.cfg.in_dim]
            y = y[...,:self.cfg.out_dim]
            all_states, encoded_state, _, _, _  = self.pretrained_encoder.encode(x, mask_s=0, mask_f=0)
            if self.cfg.backbone == 'dcrnn':
                output = self.decoder(all_states)
            elif self.cfg.backbone == 'gwnet' or self.cfg.backbone == 'mtgnn':
                output = self.decoder(encoded_state)
            else:
                output = self.decoder(encoded_state.unsqueeze(1))
            output = self.scaler.inverse_transform(output)  # [B, Tout, N, C=1]
            all_preds.append(output)
            all_targets.append(y[...,:self.cfg.out_dim])
        
        all_preds = torch.cat(all_preds, dim=0).squeeze()           # [all_sample, T_out, N]
        all_targets = torch.cat(all_targets, dim=0).squeeze()       # [all_sample, T_out, N]
        
        # horizon-wise evaluation
        metrics = metric(all_preds, all_targets, dim=(0, 2))        # [T_out]

        head = np.array(['metric'])
        for k in range(1, self.out_horizon + 1):
            head = np.append(head, [f'{k}'])
        head = np.append(head, ['average'])

        log = np.zeros([3, self.out_horizon])
        m_names = []

        for idx, (k, v) in enumerate(metrics.items()):
            m_names.append(k)
            log[idx] = metrics[k]

        m_names = np.expand_dims(m_names, axis=1)
        avg = np.mean(log, axis=1, keepdims=True)
        log = np.concatenate([m_names, log, avg], axis=1)           # [3, 1+T+1]

        print_log = 'Average Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:4f}'
        print_log_specific = 'MAE at 15 min: {:.4f}, MAE at 30 min: {:.4f}, MAE at 60 min: {:4f}'
        print(print_log.format(avg[0,0], avg[1,0], avg[2,0]))
        print(print_log_specific.format(metrics['mae'][2], metrics['mae'][5], metrics['mae'][11]))

        save_csv_log(self.cfg, head, log, is_create=True, file_property='result', file_name='result')

    def plot(self):
        plt.figure()
        plt.plot(self.train_loss_list, 'r', label='Train loss')
        plt.plot(self.valid_loss_list, 'g', label='Val loss')
        plt.legend()
        plt.savefig(os.path.join(self.cfg.vis_dir, self.cfg.id + '_ft.png'))
        plt.close()
