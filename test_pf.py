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
    parser.add_argument('--cfg', default='d03')
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
    
    # NOTE: do not need to load weights in test mode
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

    ## load best
    file_name = 'ckpt_' + cfg.id + '_ft_best.pth.tar'
    finetuner.test_load(os.path.join(cfg.model_dir, file_name))

    finetuner.test()

if __name__ == '__main__':
    main()