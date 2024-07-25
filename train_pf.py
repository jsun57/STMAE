"""2-stage training: pretrain + finetune pipeline for mask learning on spatio-temporal graphs: traffic forecasting task"""

import os
import sys
import copy
import numpy as np
import torch
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from utils import *

from data_utils.dataset_utils import load_adj
from models.load_models import get_model, get_decoder
from models.pretrainer import PreTrainer
from models.finetuner import FineTuner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='train') # mode can be pt: pretrain, ft: finetune, or train: do both
    parser.add_argument('--load_pt', action='store_true', default=False)  # for continuous training in pretraining stage
    parser.add_argument('--load_ft', action='store_true', default=False)  # for continuous training in finetuning stage
    parser.add_argument('--load_best', action='store_true', default=False)  # for continuous training in finetuning stage
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--d', type=str, default=None)
    args = parser.parse_args()

    """setup"""
    cfg = Config(args.cfg, args.d)
    set_global_seed(args.seed)
    dtype = torch.float32 if cfg.dtype == 'float32' else torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    """data"""
    if cfg.dataset == 'pems_03' or cfg.dataset == 'pems_04' or cfg.dataset == 'pems_07' or cfg.dataset == 'pems_08':
        dataset_name = cfg.dataset
    else:
        raise NotImplementedError
    _, _, raw_adj = load_adj(cfg.dataset, cfg.adj_type)

    """parameter"""
    if cfg.dataset == 'pems_03':
        cfg.num_nodes = 358
        cfg.node_emb_dim = 10
    elif cfg.dataset == 'pems_04':
        cfg.num_nodes = 307
        cfg.node_emb_dim = 10
    elif cfg.dataset == 'pems_07':
        cfg.num_nodes = 883
        cfg.node_emb_dim = 10
    elif cfg.dataset == 'pems_08':
        cfg.num_nodes = 170
        cfg.node_emb_dim = 2

    """model"""
    model = get_model(cfg, adj=raw_adj, device=device).to(dtype).to(device)

    """pretrainer setup"""
    assert cfg.pipeline == 'pretrain'
    pretrainer = PreTrainer(
        dataset=dataset_name,
        model=model,
        adj=raw_adj,
        cfg=cfg,
        batch_size=cfg.batch_size,
        pt_learning_rate=cfg.pt_lr,
        pt_weight_decay=cfg.pt_wd,
        opt_name=cfg.opt_name,
    )

    """pretrainer load && train"""
    pretrain_start_epoch = 0
    best_pretrain_loss = float('inf')
    wait = 0
    print(">>> pretrainer on:", device)
    print(">>> pretrainer params: {:.2f}M".format(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))

    if args.load_pt:
        # For continuous pretraining / finetuning
        if args.load_best:
            file_name = 'ckpt_' + cfg.id + '_pt_best.pth.tar'
        else:
            file_name = 'ckpt_' + cfg.id + '_pt_last.pth.tar'
        pretrainer.load(os.path.join(cfg.model_dir, file_name))
        pretrain_start_epoch = pretrainer.epoch
        best_pretrain_loss = pretrainer.best_train_loss

    # Pretraining
    if args.mode != "test":
        for epoch in range(pretrain_start_epoch, cfg.train_epoch):
            ret_log = np.array([epoch + 1])
            head = np.array(['epoch'])

            lr, epoch_loss, epoch_loss_info, dt = pretrainer.train()

            ret_log = np.append(ret_log, [lr, dt, epoch_loss])
            head = np.append(head, ['lr', 'dt', 't_l'])

            for key, value in epoch_loss_info.items():
                head = np.append(head, key)
                ret_log = np.append(ret_log, value)

            log = 'Epoch: {:03d}, Pretrain Loss: {:.4f}, RW Percent: {:.2f}'
            print(log.format(epoch, epoch_loss, epoch_loss_info['rw_percent']))

            # update log file and save checkpoint
            is_create = False
            if not args.load_pt:
                if epoch == 0:
                    is_create = True
            save_csv_log(cfg, head, ret_log, is_create, file_property='log', file_name=cfg.id + '_pt_log')
            
            # patience, checkpoint, info saving
            if epoch_loss < best_pretrain_loss:
                is_best = True
                best_pretrain_loss = epoch_loss
                pretrainer.best_train_loss = best_pretrain_loss
                wait = 0
            else:
                is_best = False
                wait += 1
            file_name = ['ckpt_' + cfg.id + '_pt_best.pth.tar', 'ckpt_' + cfg.id + '_pt_last.pth.tar']
            save_ckpt(cfg, pretrainer, is_best=is_best, file_name=file_name)

            if wait == cfg.patience:
                print(">>> early stopping pretrainer...")
                break

            # plotting
            pretrainer.plot()

    # Pretraining done: load best
    file_name = 'ckpt_' + cfg.id + '_pt_best.pth.tar'
    pretrainer.load(os.path.join(cfg.model_dir, file_name))

    if args.mode == 'pt':
        print('>>> pretraining finished...')
        sys.exit()
    
    pretrained_encoder = copy.deepcopy(pretrainer.model)
    del pretrainer

    """finetuner setup"""
    decoder = get_decoder(cfg, raw_adj, device=device).to(dtype).to(device)

    loss_fn = masked_mae

    finetuner = FineTuner(
        dataset=dataset_name,
        pretrained_encoder=pretrained_encoder,
        decoder=decoder,
        adj=raw_adj,
        loss=loss_fn,                           # for now, just use same loss: mask_mae
        cfg=cfg,
        batch_size=cfg.batch_size,
        learning_rate_enc=cfg.lr_enc,
        learning_rate_dec=cfg.lr_dec,
        weight_decay_enc=cfg.wd_enc,
        weight_decay_dec=cfg.wd_dec,
        ft=cfg.ft,                              # default, finetuner finetune the model; if not, only train the decoder
        opt_name=cfg.opt_name,
    )


    """finetuner load && train"""
    finetune_start_epoch = 0
    best_finetune_loss = float('inf')
    wait = 0
    print(">>> finetuner on:", device)
    print(">>> finetuner params: {:.2f}K".format(sum(p.numel() for p in decoder.parameters()) / 1000.0))

    if args.load_ft:
        # For continuous pretraining / finetuning
        if args.load_best:
            file_name = 'ckpt_' + cfg.id + '_ft_best.pth.tar'
        else:
            file_name = 'ckpt_' + cfg.id + '_ft_last.pth.tar'        
        finetuner.load(os.path.join(cfg.model_dir, file_name))
        finetune_start_epoch = finetuner.epoch
        best_finetune_loss = finetuner.best_valid_loss

    # Finetuning
    if args.mode != "test":
        for epoch in range(finetune_start_epoch, cfg.ft_epoch):
            ret_log = np.array([epoch + 1])
            head = np.array(['epoch'])

            en_lr, de_lr, epoch_loss, epoch_loss_info, dt = finetuner.train()

            ret_log = np.append(ret_log, [en_lr, de_lr, dt, epoch_loss])
            head = np.append(head, ['en_lr', 'de_lr', 'dt', 't_l'])

            for key, value in epoch_loss_info.items():
                head = np.append(head, key)
                ret_log = np.append(ret_log, value)

            # validation
            valid_loss_info = finetuner.valid()

            log = 'Epoch: {:03d}, Finetune Train Loss: {:.4f}, Valid Loss: {:.4f}'
            print(log.format(epoch, epoch_loss, valid_loss_info['mae']))

            # add validation logging
            for key, value in valid_loss_info.items():
                head = np.append(head, key)
                ret_log = np.append(ret_log, value)

            # update log file and save checkpoint
            is_create = False
            if not args.load_ft:
                if epoch == 0:
                    is_create = True
            save_csv_log(cfg, head, ret_log, is_create, file_property='log', file_name=cfg.id + '_ft_log')
            
            # patience, checkpoint, info saving
            if valid_loss_info['mae'] < best_finetune_loss:
                is_best = True
                best_finetune_loss = valid_loss_info['mae']
                finetuner.best_finetune_loss = best_finetune_loss
                wait = 0
            else:
                is_best = False
                wait += 1
            file_name = ['ckpt_' + cfg.id + '_ft_best.pth.tar', 'ckpt_' + cfg.id + '_ft_last.pth.tar']
            save_ckpt(cfg, finetuner, is_best=is_best, file_name=file_name)

            if wait == cfg.patience:
                print(">>> early stopping finetuner...")
                break

            # plotting
            finetuner.plot()


    print('All training end, compute final stats...')
    
    ## load best
    file_name = 'ckpt_' + cfg.id + '_ft_best.pth.tar'
    finetuner.load(os.path.join(cfg.model_dir, file_name))

    finetuner.test()

if __name__ == '__main__':
    main()