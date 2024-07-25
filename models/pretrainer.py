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


class PreTrainer(object):
    def __init__(
        self,
        dataset,
        model,
        adj,
        cfg,
        batch_size,
        pt_learning_rate,
        pt_weight_decay,
        opt_name='adam',
    ):
        super().__init__()

        self.model = model
        self.device = next(self.model.parameters()).device
        self.adj = adj  # The adjacency matrix is from https://arxiv.org/pdf/2108.11873.pdf: weighted with self_loops

        self.cfg = cfg

        self.batch_size = batch_size
        self.opt_name = opt_name
        self.learning_rate = pt_learning_rate
        self.weight_decay = pt_weight_decay

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
        self.opt = optimizer(self.model.parameters(), lr=self.learning_rate, eps=self.cfg.pt_eps, weight_decay=self.weight_decay)

        self.scheduler = get_scheduler(self.opt, policy=self.cfg.pt_sched_policy, nepoch_fix=self.cfg.pt_num_epoch_fix_lr, nepoch=self.cfg.train_epoch, \
            decay_step=self.cfg.pt_decay_step, gamma=self.cfg.pt_gamma, milestones=self.cfg.pt_milestones)
        
        self.epoch = 0
        self.train_loss_list = []
        self.best_train_loss = float('inf')
        self.batches_seen = None
        print('Pretrainer initialization done.')

    def save(self, to_save_path):
        data = {
            'epoch': self.epoch,
            'train_loss_list': self.train_loss_list,
            'best_train_loss': self.best_train_loss,
            'batches_seen': self.batches_seen,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'sched': self.scheduler.state_dict() if exists(self.scheduler) else None,
        }
        torch.save(data, to_save_path)
        return

    def load(self, to_load_path):
        if self.cfg.backbone == 'dcrnn':
            self.model(torch.rand(1, self.in_horizon, self.cfg.num_nodes, self.cfg.in_dim).to(self.device).to(self.dtype))
            optimizer = Adam if self.opt_name == 'adam' else AdamW
            self.opt = optimizer(self.model.parameters(), lr=self.learning_rate, eps=self.cfg.pt_eps, weight_decay=self.weight_decay)
            self.scheduler = get_scheduler(self.opt, policy=self.cfg.pt_sched_policy, nepoch_fix=self.cfg.pt_num_epoch_fix_lr, nepoch=self.cfg.train_epoch, \
                decay_step=self.cfg.pt_decay_step, gamma=self.cfg.pt_gamma, milestones=self.cfg.pt_milestones)
        device = self.device
        data = torch.load(to_load_path, map_location=device)
        self.epoch = data['epoch']
        self.train_loss_list = data['train_loss_list']
        self.best_train_loss = data['best_train_loss']
        self.batches_seen = data['batches_seen']
        self.model.load_state_dict(data['model'])
        self.opt.load_state_dict(data['opt'])
        if exists(data['sched']):
            self.scheduler.load_state_dict(data['sched'])  
        else: self.scheduler = None
        print(">>> finish loading pretrainer model ckpt from path '{}'".format(to_load_path))
        return

    def train(self):
        """
        pretrain model for one epoch
        """ 
        self.model.train()
        t_s = time.time()
        epoch_loss = 0.
        epoch_iter = 0
        epoch_loss_info = {}
        self.batches_seen = self.dataloader['train_loader'].num_batch * self.epoch    # this is for dcrnn specific training
        model_configs = {**self.cfg.mask_specs, **self.cfg.loss_specs}
        self.dataloader['train_loader'].shuffle()

        # TODO: epoch-wise mask: generate f_mask, s_mask here instead
        if self.cfg.epoch_wise_mask:
            x_holder = torch.rand(self.batch_size, self.in_horizon, self.cfg.num_nodes, self.cfg.in_dim).to(self.device)
            s_mask, _ = self.model.structure_masking(x_holder, mask_ratio=self.cfg.mask_s, mask=None, **model_configs)
            _, f_mask = self.model.feature_masking(x_holder, mask_ratio=self.cfg.mask_f, mask=None, **model_configs)
            model_configs = {**model_configs, 's_mask': s_mask, 'f_mask': f_mask}

        for idx, (x, _) in enumerate(self.dataloader['train_loader'].get_iterator()):
            x = x[...,:self.cfg.in_dim]
            loss, loss_info = self.model(x, batches_seen=self.batches_seen, **model_configs)
            for key, value in loss_info.items():
                if key not in epoch_loss_info:
                    epoch_loss_info[key] = value
                else:
                    epoch_loss_info[key] += value

            if self.batches_seen == 0 and self.cfg.backbone == 'dcrnn':   # dcrnn only
                optimizer = Adam if self.opt_name == 'adam' else AdamW
                self.opt = optimizer(self.model.parameters(), lr=self.learning_rate, eps=self.cfg.pt_eps, weight_decay=self.weight_decay)
                self.scheduler = get_scheduler(self.opt, policy=self.cfg.pt_sched_policy, nepoch_fix=self.cfg.pt_num_epoch_fix_lr, nepoch=self.cfg.train_epoch, \
                decay_step=self.cfg.pt_decay_step, gamma=self.cfg.pt_gamma, milestones=self.cfg.pt_milestones)

            self.opt.zero_grad()
            loss.backward()
            if exists(self.cfg.pt_clip_grad) and self.cfg.pt_clip_grad != 'None':
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.pt_clip_grad)
            self.opt.step()
            epoch_loss += loss.item()
            self.batches_seen += 1
            epoch_iter += 1
        if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and exists(self.scheduler):
            self.scheduler.step()
        
        self.epoch += 1
        epoch_loss /= epoch_iter
        for key, value in epoch_loss_info.items():
            epoch_loss_info[key] /= epoch_iter
        lr = self.opt.param_groups[0]['lr']
        dt = time.time() - t_s
        self.train_loss_list.append(epoch_loss)
        return lr, epoch_loss, epoch_loss_info, dt 
    
    def plot(self):
        plt.figure()
        plt.plot(self.train_loss_list, 'r', label='Train loss')
        plt.legend()
        plt.savefig(os.path.join(self.cfg.vis_dir, self.cfg.id + '_pt.png'))
        plt.close()
