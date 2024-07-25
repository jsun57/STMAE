import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange, repeat
from .pf_utils import mask_path

"""
This AGCRN is used for 2-stage training scheme: Pretrain + Finetune
"""

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    
    def apply_s_mask(self, support, s_mask, normalize=False):
        """
        TODO: structrual masking here: for AGCRN, batch-wise masking is the same
        support: [N, N] row stochastic matrix
        NOTE: mask: 1 is keep, 0 is masked
        TODO: should we fix this per training iteration, for better loss decoding?
        Current: L * T * num_rnn_units dynamic masking
        """
        N, _ = support.shape
        
        support_masked = support * s_mask
        if normalize:
            support_masked = F.softmax(support_masked, dim=1)

        return support_masked

    def forward(self, x, node_embeddings, s_mask, normalize=False):
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1) # [N, N]: row sum is 1

        ### perform graph structure masking ###
        supports = self.apply_s_mask(supports, s_mask, normalize)
        #######################################
        
        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings, s_mask, normalize):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, s_mask, normalize))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, s_mask, normalize))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, s_mask, normalize):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, s_mask, normalize)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, act=torch.sigmoid, with_proj=False, hid_dim=None):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.proj = nn.Identity() if not with_proj else nn.Conv1d(hid_dim, hid_dim, kernel_size=1)

    def forward(self, z):
        """
        z: [B, N, F]
        """
        z = self.dropout(z)
        z = self.proj(z.permute(0, 2, 1)).permute(0, 2, 1)
        adj = self.act(torch.bmm(z, z.permute(0, 2, 1)))
        return adj



class AGCRN(nn.Module):
    def __init__(
        self, 
        num_nodes, 
        embed_dim, 
        in_dim, 
        out_dim, 
        rnn_units, 
        num_layers, 
        cheb_k,
        in_horizon,
        out_horizon,
        stru_dec_drop,
        stru_dec_proj,
    ):
        super(AGCRN, self).__init__()
        self.num_nodes = num_nodes
        self.output_dim = out_dim
        self.hidden_dim = rnn_units
        self.in_horizon = in_horizon
        self.out_horizon = out_horizon
        self.stru_dec_drop = stru_dec_drop

        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)

        self.mask_token = nn.Parameter(torch.randn(rnn_units), requires_grad=True)

        self.encoder = AVWDCRNN(num_nodes, rnn_units, rnn_units, cheb_k, embed_dim, num_layers)

        # TODO: more complex structure decoder
        self.structure_decoder = InnerProductDecoder(stru_dec_drop, act=lambda x: x, with_proj=stru_dec_proj, hid_dim=rnn_units)

        # TODO: more complex feature decoder
        self.feature_decoder = nn.Conv2d(1, in_horizon, kernel_size=(1, rnn_units), bias=True)
        
        self.to_feat_embedding = nn.Linear(in_dim, rnn_units)

        self.init_parameters()

    def init_parameters(self):
        """follow STGCL way"""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)


    def get_support(self):
        """
        get current support: [N, N]: between 0 to 1 element-wise
        """
        support = torch.sigmoid(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1)))
        return support


    def feature_masking(self, x, mask_ratio, mask_f_strategy='uniform', mask=None, *args, **kwargs):
        """
        masking according to strategy: per sample masking on time axis
        x: [B, Tin, N, Din=1]
        NOTE: mask: 1 is keep, 0 is masked
        """
        B, T, N, D = x.shape

        if mask is not None:
            x_masked = x * mask.unsqueeze(-1)
            return x_masked, mask

        if mask_ratio == 0:
            mask = torch.ones([B, T, N], device=x.device)
            x_masked = x
            return x_masked, mask

        if mask_f_strategy == 'uniform':
            # simplest masking: same per batch, same area per node
            len_discard = round(T * mask_ratio)
            
            noise = torch.rand(B, T, N, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # generate binary mask
            mask = torch.ones([B, T, N], device=x.device)
            mask[:, :len_discard, :] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)

            # mask by zero 
            x_masked = x * mask.unsqueeze(-1)

        elif mask_f_strategy == 'patch_uniform':
            
            patch_length = kwargs.get('patch_length', 1)    # default is 1: collaspe to uniform
            assert patch_length <= T / 2 and T % patch_length == 0, 'patch_length need to be smaller than sequence length and dividable.'
            num_patches = T // patch_length
            num_masked_patches = round(num_patches * mask_ratio)
            
            # Initialize the mask with all ones: mask on T dim
            mask = torch.ones([B, T, N], device=x.device)  
            
            # Randomly select patches to be masked
            masked_indices = torch.randperm(num_patches)[:num_masked_patches]
            
            # Generate the indices to mask
            start_indices = (masked_indices * patch_length).to(dtype=torch.long, device=x.device)
            end_indices = (start_indices + patch_length).to(dtype=torch.long, device=x.device)

            ranges = torch.stack([torch.arange(start, end) for start, end in zip(start_indices, end_indices)])
            all_indices = torch.flatten(ranges).to(x.device)
            mask.scatter_(1, repeat(all_indices, 't -> b t n', b=B, n=N), 0)

            # mask by zero 
            x_masked = x * mask.unsqueeze(-1)

        elif mask_f_strategy == 'rw':
            # TODO  
            pass

        return x_masked, mask


    def structure_masking(self, x, mask_ratio, mask_s_strategy='uniform', mask=None, *args, **kwargs):
        """
        TODO: structrual masking here: for AGCRN, batch-wise masking is the same
        support: [N, N] row stochastic matrix
        NOTE: mask: 1 is keep, 0 is masked
        TODO: should we fix this per training iteration, for better loss decoding?
        Current: L * T * num_rnn_units dynamic masking
        """
        B, _, N, _ = x.shape

        rw_percent = -1

        if mask is not None:
            return mask, rw_percent

        if mask_ratio == 0:
            return torch.ones(N, N, device=x.device), rw_percent

        if mask_s_strategy == 'uniform':
            # simplest masking: same per batch, same area per node
            num_discard = round(N * N * mask_ratio)
            
            orig_indices = torch.nonzero(torch.ones_like(self.get_support()))
            shuffled_idx = torch.randperm(N * N)
            mask_indices = shuffled_idx[:num_discard]

            mask = torch.ones(N, N, device=x.device)
            actual_mask_idx = orig_indices[mask_indices]
            mask[actual_mask_idx[:, 0], actual_mask_idx[:, 1]] = 0

        elif mask_s_strategy == 'uniform_post_ce':

            with_negative = kwargs.get('with_negative', False)

            current_support = self.get_support()    # [N, N]
            binaried_support = (current_support > 0.5).to(current_support.dtype)

            num_edges = torch.count_nonzero(binaried_support).item()
            num_discard = round(num_edges * mask_ratio)

            orig_indices = torch.nonzero(binaried_support)
            shuffled_idx = torch.randperm(num_edges)
            mask_indices = shuffled_idx[:num_discard]

            mask = torch.ones(N, N, device=x.device)
            actual_mask_idx = orig_indices[mask_indices]
            mask[actual_mask_idx[:, 0], actual_mask_idx[:, 1]] = 0

            if with_negative:
                # negative edge selection
                ne_edge_indices = torch.nonzero(abs(binaried_support - 1))
                num_non_edges = ne_edge_indices.size(0)
                shuffled_idx = torch.randperm(num_non_edges)
                mask_indices = shuffled_idx[:num_discard]

                actual_mask_idx = ne_edge_indices[mask_indices]
                mask[actual_mask_idx[:, 0], actual_mask_idx[:, 1]] = 0

        elif mask_s_strategy == 'rw':
            
            with_negative = kwargs.get('with_negative', False)

            # random-walk based path masking strategy
            current_support = self.get_support()    # [N, N]
            binaried_support = (current_support > 0.5).to(current_support.dtype)
            num_edges = torch.count_nonzero(binaried_support).item()

            walks_per_node = kwargs.get('walks_per_node', 1)
            walk_length = kwargs.get('walk_length', 3)
            start = kwargs.get('start', 'node')
            p = kwargs.get('p', 1.0)
            q = kwargs.get('q', 1.0)
            masked_edge_index, num_discard = mask_path(binaried_support, mask_ratio, walks_per_node=walks_per_node, \
            walk_length=walk_length, start=start, p=p, q=q)

            mask = torch.ones_like(binaried_support)
            mask[masked_edge_index[0, :], masked_edge_index[1, :]] = 0

            if with_negative:
                # negative edge selection
                ne_edge_indices = torch.nonzero(abs(binaried_support - 1))
                num_non_edges = ne_edge_indices.size(0)
                shuffled_idx = torch.randperm(num_non_edges)
                mask_indices = shuffled_idx[:num_discard]
                
                actual_mask_idx = ne_edge_indices[mask_indices]
                mask[actual_mask_idx[:, 0], actual_mask_idx[:, 1]] = 0

        elif mask_s_strategy == 'rw_fill':
            
            goal_discard = round(N * N * mask_ratio)

            # STEP1: random-walk based path masking: fully connected graph 
            binaried_support = torch.ones_like(self.get_support())
            walks_per_node = kwargs.get('walks_per_node', 10)
            walk_length = kwargs.get('walk_length', 20)
            start = kwargs.get('start', 'node')
            p = kwargs.get('p', 1.0)
            q = kwargs.get('q', 1.0)
            masked_edge_index, num_discard = mask_path(binaried_support, mask_ratio=mask_ratio, walks_per_node=walks_per_node, \
            walk_length=walk_length, start=start, p=p, q=q) # [2, num_masked]
            
            rw_percent = num_discard / goal_discard
            
            # STEP2: if more, discard; else, uniform add
            mask = torch.ones_like(binaried_support)
            if goal_discard > num_discard:
                mask[masked_edge_index[0, :], masked_edge_index[1, :]] = 0
                # uniform masking
                remain_discard = goal_discard - num_discard
                remain_idx = torch.nonzero(binaried_support * mask)
                shuffled_idx = torch.randperm(remain_idx.size(0))
                mask_indices = shuffled_idx[:remain_discard]
                remain_actual_mask_idx = remain_idx[mask_indices]
                mask[remain_actual_mask_idx[:, 0], remain_actual_mask_idx[:, 1]] = 0
            else:
                masked_edge_index = masked_edge_index[:,:goal_discard]
                mask[masked_edge_index[0, :], masked_edge_index[1, :]] = 0

        else:
            raise NotImplementedError

        return mask, rw_percent


    def encode(self, x, mask_s=0, mask_f=0, mask_s_strategy='uniform', mask_f_strategy='uniform', *args, **kwargs):
        """
        input: [B, Tin, N, Din]
        output: [B, Tin, N, F], [B, N, F]
        """
        B, T, N, _ = x.shape
        init_state = self.encoder.init_hidden(x.shape[0])  # [L, B, N, F]

        raw_x = x[...,:1].clone()
        
        # random_masking for now: TODO: other strategies
        x_masked, f_mask = self.feature_masking(raw_x, mask_f, mask_f_strategy, mask=kwargs.get('f_mask', None), *args, **kwargs)

        # feature embedding with mask embedding
        embed_x = self.to_feat_embedding(x_masked)  # [B, T, N, F]
        embed_x = embed_x * f_mask.unsqueeze(-1) + repeat(self.mask_token, 'd -> b t n d', b=B, t=T, n=N) * (1 - f_mask.unsqueeze(-1))

        # structure mask:
        s_mask, rw_percent = self.structure_masking(raw_x, mask_s, mask_s_strategy, mask=kwargs.get('s_mask', None), *args, **kwargs)

        # encode
        normalize = kwargs.get('normalize', False)
        embedding, _ = self.encoder(embed_x, init_state, self.node_embeddings, s_mask, normalize=normalize)
        summary = embedding[:, -1:, :, :].squeeze(1)

        return embedding, summary, f_mask, s_mask, rw_percent


    def decode_structure(self, summary):
        """
        summary: [B, N, H]
        output: [B, N, N], between (0, 1)
        TODO: add inter projecter? Use node_embedding instead of summary?
        """
        return self.structure_decoder(summary)


    def decode_feature(self, summary):
        """
        summary: [B, N, H]
        output: [B, Tin, N, Din]
        TODO: add inter projecter? Use embedding instead of summary?
        """
        return self.feature_decoder(summary.unsqueeze(1))
        

    def forward_s_loss(self, target, pred, s_mask, l_type='reg_l2'):
        """
        target (support): [N, N]
        pred: [B, N, N]
        s_mask: [N, N]
        """
        # structure loss
        B = pred.shape[0]
        if l_type == 'reg_l2':
            target = repeat(target, 'm n -> b m n', b=B)
            si_mask = repeat(abs(s_mask - 1), 'm n -> b m n', b=B)
            if torch.sum(si_mask) != 0:
                diff = (torch.flatten(pred) - torch.flatten(target)) ** 2 * torch.flatten(si_mask)
                loss = torch.sum(diff) / torch.sum(si_mask)
            else:
                # do not mask: this is for ablation
                diff = (torch.flatten(pred) - torch.flatten(target)) ** 2
                loss = torch.mean(diff)
        
        elif l_type == 'reg_l1':
            target = repeat(target, 'm n -> b m n', b=B)
            si_mask = repeat(abs(s_mask - 1), 'm n -> b m n', b=B)
            if torch.sum(si_mask) != 0:
                diff = torch.abs(torch.flatten(pred) - torch.flatten(target)) * torch.flatten(si_mask)
                loss = torch.sum(diff) / torch.sum(si_mask)
            else:
                # do not mask: this is for ablation
                diff = torch.abs(torch.flatten(pred) - torch.flatten(target))
                loss = torch.mean(diff)

        elif l_type == 'cls_ce':
            target = (target > 0.5).to(pred.dtype)
            target = repeat(target, 'm n -> b m n', b=B)
            si_mask = repeat(abs(s_mask - 1), 'm n -> b m n', b=B)
            if torch.sum(si_mask) != 0:
                diff = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * si_mask
                loss = torch.sum(diff) / torch.sum(si_mask)
            else:
                diff = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
                loss = torch.mean(diff)

        elif l_type == 'cls_boost':
            target = torch.ones_like(target)
            target = repeat(target, 'm n -> b m n', b=B)
            si_mask = repeat(abs(s_mask - 1), 'm n -> b m n', b=B)
            if torch.sum(si_mask) != 0:
                diff = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * si_mask
                loss = torch.sum(diff) / torch.sum(si_mask)
            else:
                diff = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
                loss = torch.mean(diff)

        else:
            raise NotImplementedError

        return loss


    def forward_f_loss(self, target, pred, f_mask, l_type='reg_l2'):
        """
        target (support): [B, Tin, N, C=2]?
        pred: [B, Tin, N, C=1]
        f_mask: [B, Tin, N]
        """
        # TODO: feature recon. SCE? https://arxiv.org/pdf/2205.10803.pdf
        target = target[...,:1]

        # feature loss
        B, T, N, _ = pred.shape
        if l_type == 'reg_l2':
            target, pred = target.squeeze(-1), pred.squeeze(-1)
            fi_mask = abs(f_mask - 1)
            if torch.sum(fi_mask) != 0:
                diff = (torch.flatten(pred) - torch.flatten(target)) ** 2 * torch.flatten(fi_mask)
                loss = torch.sum(diff) / torch.sum(fi_mask)
            else:
                # do not mask: this is for ablation
                diff = (torch.flatten(pred) - torch.flatten(target)) ** 2
                loss = torch.mean(diff)
        
        elif l_type == 'reg_l1':
            target, pred = target.squeeze(-1), pred.squeeze(-1)
            fi_mask = abs(f_mask - 1)
            if torch.sum(fi_mask) != 0:
                diff = torch.abs(torch.flatten(pred) - torch.flatten(target)) * torch.flatten(fi_mask)
                loss = torch.sum(diff) / torch.sum(fi_mask)
            else:
                # do not mask: this is for ablation
                diff = torch.abs(torch.flatten(pred) - torch.flatten(target))
                loss = torch.mean(diff)
        else:
            raise NotImplementedError

        return loss


    def forward(self, x, mask_s=0, mask_f=0, mask_s_strategy='uniform', mask_f_strategy='uniform', *args, **kwargs):
        """
        TODO: forward should output loss only 
        """
        # f_mask: [B, Tin, N]; s_mask: [N, N]
        embedding, summary, f_mask, s_mask, rw_percent = self.encode(x, mask_s, mask_f, mask_s_strategy, mask_f_strategy, *args, **kwargs)

        recon_s = self.decode_structure(summary)    # [B, N, N]
        recon_f = self.decode_feature(summary)      # [B, T, N, C=1]

        sl_type = kwargs.get('sl_type', 'reg_l2')
        fl_type = kwargs.get('fl_type', 'reg_l2')
        s_loss = self.forward_s_loss(self.get_support(), recon_s, s_mask, sl_type)
        f_loss = self.forward_f_loss(x, recon_f, f_mask, fl_type)

        s_weight = kwargs.get('sl_weight', 1.0)
        f_weight = kwargs.get('fl_weight', 1.0)
        loss = s_weight * s_loss + f_weight * f_loss
        
        loss_info = {'s_loss': s_loss.item(), 'f_loss': f_loss.item(), 'loss': loss.item(), 'rw_percent': rw_percent}
        return loss, loss_info


class AGCRN_Decoder(nn.Module):
    """
    agcrn decoder: exactly follow STGCL
    """
    def __init__(self, out_dim, rnn_units, horizon, de_mlp=False):
        super(AGCRN_Decoder, self).__init__()
        self.de_mlp = de_mlp
        if not self.de_mlp:
            self.end_conv = nn.Conv2d(1, horizon * out_dim, kernel_size=(1, rnn_units), bias=True)
        else:
            self.end_conv_1 = nn.Conv2d(rnn_units, rnn_units * 8, kernel_size=(1, 1), bias=True)
            self.end_conv_2 = nn.Conv2d(rnn_units * 8, horizon * out_dim, kernel_size=(1, 1), bias=True)
        
    def forward(self, input):
        if not self.de_mlp:
            x = self.end_conv(input)
        else:
            x = input.transpose(1,3)
            x = F.relu(self.end_conv_1(x))
            x = self.end_conv_2(x)
        return x



if __name__ == "__main__":
    B = 2
    Tin = 12
    N = 2
    D = 1
    Tout = 12


    torch.manual_seed(0)
    x = torch.randn(B, Tin, N, D)

    mdl= AGCRN(
        num_nodes=N, 
        embed_dim=B, 
        in_dim=D,
        out_dim=D, 
        rnn_units=2, 
        num_layers=4, 
        cheb_k=2, 
        in_horizon=Tin,
        out_horizon=Tout,
        stru_dec_drop=0.1,
    ) 

    decoder = AGCRN_Decoder(
        out_dim=1, rnn_units=2, horizon=Tout, de_mlp=True
    )
    _, encoded, _, _ = mdl.encode(x, mask_f_strategy='patch_uniform', mask_f=0.75, patch_length=3)
    exit()
    decoded = decoder(encoded.unsqueeze(1))
    print(encoded.shape)
    print(decoded.shape)