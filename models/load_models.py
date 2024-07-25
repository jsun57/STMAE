import torch
from models.pfmodels.pf_agcrn import AGCRN, AGCRN_Decoder
from models.pfmodels.pf_dcrnn import DCRNN, DCRNN_Decoder
from models.pfmodels.pf_mtgnn import MTGNN, MTGNN_Decoder

def get_model(cfg, adj=None, device=torch.device('cpu')):
    model_name = cfg.backbone
    if model_name == 'agcrn':
        model = AGCRN(
            num_nodes=cfg.num_nodes, 
            embed_dim=cfg.node_emb_dim, 
            in_dim=cfg.in_dim,
            out_dim=cfg.out_dim, 
            rnn_units=cfg.rnn_dim, 
            num_layers=cfg.num_layers, 
            cheb_k=cfg.cheb_k, 
            in_horizon=cfg.in_horizon,
            out_horizon=cfg.out_horizon,
            stru_dec_drop=cfg.stru_dec_drop,
            stru_dec_proj=cfg.stru_dec_proj,
        )
    elif model_name == 'dcrnn':
        model = DCRNN(
            adj_mx=adj,
            num_nodes=cfg.num_nodes,
            num_rnn_layers=cfg.num_layers,
            rnn_units=cfg.rnn_dim,
            input_dim=cfg.in_dim,
            seq_len=cfg.in_horizon,
            max_diffusion_step=cfg.max_diffusion_step,
            cl_decay_steps=cfg.cl_decay_steps,
            filter_type=cfg.filter_type,
            stru_dec_drop=cfg.stru_dec_drop,
            stru_dec_proj=cfg.stru_dec_proj,
            device=device,
        )
    elif model_name == 'mtgnn':
        model = MTGNN(
            gcn_true=True, 
            buildA_true=True, 
            gcn_depth=cfg.gcn_depth, 
            num_nodes=cfg.num_nodes, 
            device=device, 
            predefined_A=adj, 
            static_feat=None, 
            dropout=0.3, 
            subgraph_size=20, 
            node_dim=40, 
            dilation_exponential=1, 
            conv_channels=cfg.rnn_dim,
            residual_channels=cfg.rnn_dim,
            skip_channels=cfg.rnn_dim*2, 
            end_channels=cfg.rnn_dim*4,
            seq_length=cfg.in_horizon,
            in_dim=cfg.in_dim, 
            out_dim=cfg.out_dim,
            horizon=cfg.out_horizon,
            layers=cfg.num_layers,
            propalpha=0.05, 
            tanhalpha=3, 
            layer_norm_affline=True,
            stru_dec_drop=cfg.stru_dec_drop,
            stru_dec_proj=cfg.stru_dec_proj,
        )
    else:
        raise NotImplementedError
    return model


def get_decoder(cfg, adj=None, device=torch.device('cpu')):
    model_name = cfg.backbone
    if model_name == 'agcrn':
        model = AGCRN_Decoder(
            out_dim=cfg.out_dim, 
            rnn_units=cfg.rnn_dim, 
            horizon=cfg.out_horizon, 
            de_mlp=cfg.de_mlp,
        )
    elif model_name == 'dcrnn':
        model = DCRNN_Decoder(
            adj_mx=adj,
            num_nodes=cfg.num_nodes,
            num_rnn_layers=cfg.num_layers,
            rnn_units=cfg.rnn_dim,
            output_dim=cfg.out_dim,
            horizon=cfg.out_horizon,
            max_diffusion_step=cfg.max_diffusion_step,
            cl_decay_steps=cfg.cl_decay_steps,
            filter_type=cfg.filter_type,
            use_curriculum_learning=cfg.use_curriculum_learning,
            de_mlp=cfg.de_mlp,
            device=device,
        )
    elif model_name == 'gwnet':
        model = GWNET_Decoder(
            skip_channels=cfg.rnn_dim*8, 
            end_channels=cfg.rnn_dim*16, 
            horizon=cfg.out_horizon, 
            de_mlp=cfg.de_mlp,
        )
    elif model_name == 'mtgnn':
        model = MTGNN_Decoder(
            skip_channels=cfg.rnn_dim*2, 
            end_channels=cfg.rnn_dim*4, 
            horizon=cfg.out_horizon, 
            de_mlp=cfg.de_mlp,
        )
    else:
        raise NotImplementedError
    return model
