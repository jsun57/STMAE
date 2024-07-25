import os
import torch
import torch.nn as nn
import ruamel.yaml
yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
yaml.explicit_start = True

class Config:

    def __init__(self, cfg_id, date=None):
        self.id = cfg_id
        if date is not None:
            cfg_name = './cfg/%s/%s.yml' % (date, cfg_id)
        else:
            cfg_name = './cfg/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.load(open(cfg_name, 'r'))

        # create dirs
        self.base_dir = 'results'

        self.cfg_dir = '%s/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.vis_dir = '%s/vis' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        # # common
        self.pipeline = cfg.get('pipeline', 'pretrain')
        self.backbone = cfg.get('backbone', 'agcrn')
        self.in_horizon = cfg.get('in_horizon', 12)
        self.out_horizon = cfg.get('out_horizon', 12)
        self.dtype = cfg.get('dtype', 'float32')
        self.adj_type = cfg.get('adj_type', None)

        # # backbone specific configs
        self.backbone_specs = cfg.get('backbone_specs', dict())
        self.in_dim = self.backbone_specs.get('in_dim', 1)
        self.out_dim = self.backbone_specs.get('out_dim', 1)
        self.stru_dec_drop = self.backbone_specs.get('stru_dec_drop', 0)
        self.stru_dec_proj = self.backbone_specs.get('stru_dec_proj', False)
        # agcrn
        self.rnn_dim = self.backbone_specs.get('rnn_dim', 64)
        self.proj_dim = self.backbone_specs.get('proj_dim', 32)
        self.num_layers = self.backbone_specs.get('num_layers', 2)
        self.cheb_k = self.backbone_specs.get('cheb_k', 2)
        # dcrnn
        self.max_diffusion_step = self.backbone_specs.get('max_diffusion_step', 2)
        self.cl_decay_steps = self.backbone_specs.get('cl_decay_steps', 2000)
        self.filter_type = self.backbone_specs.get('filter_type', 'dual_random_walk')
        self.use_curriculum_learning = self.backbone_specs.get('use_curriculum_learning', True)
        # mtgnn
        self.gcn_depth = self.backbone_specs.get('gcn_depth', 2)
        
        # # data
        self.data_specs = cfg.get('data_specs', dict())
        self.dataset = self.data_specs.get('dataset', 'pems_04')

        # # mask
        self.mask_specs = cfg.get('mask_specs', dict())
        self.epoch_wise_mask = self.mask_specs.get('epoch_wise_mask', False)
        self.mask_s = self.mask_specs.get('mask_s', 0)
        self.mask_f = self.mask_specs.get('mask_f', 0)
        self.mask_s_strategy = self.mask_specs.get('mask_s_strategy', 'uniform')
        self.mask_f_strategy = self.mask_specs.get('mask_f_strategy', 'uniform')
        self.normalize = self.mask_specs.get('normalize', False)
        self.patch_length = self.mask_specs.get('patch_length', 3)
        self.with_negative = self.mask_specs.get('with_negative', False)
        # gwnet specific
        self.gwnet_mask_type = self.mask_specs.get('gwnet_mask_type', 'all')  
        # mtgnn specific
        self.pre_mask = self.mask_specs.get('pre_mask', False)  
        # rw
        self.walks_per_node = self.mask_specs.get('walks_per_node', 1)
        self.walk_length = self.mask_specs.get('walk_length', 3)
        self.start = self.mask_specs.get('start', 'node')
        self.p = self.mask_specs.get('p', 1.0)
        self.q = self.mask_specs.get('q', 1.0)

        # # learning
        self.learn_specs = cfg.get('learn_specs', dict())
        # pretrain
        self.opt_name = cfg.get('opt_name', 'adam')        
        self.pt_lr = self.learn_specs.get('pt_lr', 1e-3)
        self.pt_wd = self.learn_specs.get('pt_wd', 0)
        self.pt_eps = self.learn_specs.get('pt_eps', 1e-8)
        self.batch_size = self.learn_specs.get('batch_size', 64)
        self.train_epoch = self.learn_specs.get('train_epoch', 100)
        self.pt_sched_policy = self.learn_specs.get('pt_sched_policy', 'None')
        self.pt_num_epoch_fix_lr = self.learn_specs.get('pt_num_epoch_fix_lr', 0)
        self.pt_decay_step = self.learn_specs.get('pt_decay_step', 0)
        self.pt_gamma = self.learn_specs.get('pt_gamma', 0)
        self.pt_milestones = self.learn_specs.get('pt_milestones', [])
        self.pt_clip_grad = self.learn_specs.get('pt_clip_grad', 'None')
        # finetune
        self.de_mlp = self.learn_specs.get('de_mlp', False)  # if not, only finetune decoder
        self.ft = self.learn_specs.get('ft', True)  # if not, only finetune decoder
        self.lr_enc = self.learn_specs.get('lr_enc', 1e-4)
        self.lr_dec = self.learn_specs.get('lr_dec', 1e-3)
        self.wd_enc = self.learn_specs.get('wd_enc', 0)
        self.wd_dec = self.learn_specs.get('wd_dec', 0)
        self.eps_enc = self.learn_specs.get('eps_enc', 1e-8)
        self.eps_dec = self.learn_specs.get('eps_dec', 1e-8)
        self.ft_epoch = self.learn_specs.get('ft_epoch', 100)
        self.ft_sched_policy = self.learn_specs.get('ft_sched_policy', 'None')
        self.ft_num_epoch_fix_lr = self.learn_specs.get('ft_num_epoch_fix_lr', 0)
        self.ft_decay_step = self.learn_specs.get('ft_decay_step', 0)
        self.ft_gamma = self.learn_specs.get('ft_gamma', 0)
        self.ft_milestones = self.learn_specs.get('ft_milestones', [])
        self.ft_clip_grad = self.learn_specs.get('ft_clip_grad', 'None')
        # extra
        self.lam = self.learn_specs.get('lam', 1)
        self.patience = self.learn_specs.get('patience', -1)
        
        # # loss
        self.loss_specs = cfg.get('loss_specs', dict())
        self.sl_weight = self.loss_specs.get('sl_weight', 1.0)
        self.fl_weight = self.loss_specs.get('fl_weight', 1.0)
        self.sl_type = self.loss_specs.get('sl_type', 'reg_l2')
        self.fl_type = self.loss_specs.get('fl_type', 'reg_l2')
