import sys
sys.path.append('../')
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange, repeat
from .pf_utils import mask_path
import warnings

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()


class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho


class dy_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)
        self.lin2 = linear(c_in,c_in)


    def forward(self,x):
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj0)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)


        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2


class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx, s_mask=None):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))

        # pre_mask here 
        if s_mask is not None:
            a = a * s_mask

        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj * mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


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


class MTGNN(nn.Module):
    def __init__(
        self, 
        gcn_true, 
        buildA_true, 
        gcn_depth, 
        num_nodes, 
        device, 
        predefined_A=None, 
        static_feat=None, 
        dropout=0.3, 
        subgraph_size=20, 
        node_dim=40, 
        dilation_exponential=1, 
        conv_channels=32, 
        residual_channels=32, 
        skip_channels=64, 
        end_channels=128, 
        seq_length=12, 
        in_dim=2,
        out_dim=1,
        horizon=12, 
        layers=3, 
        propalpha=0.05, 
        tanhalpha=3, 
        layer_norm_affline=True,
        stru_dec_drop=0.0,
        stru_dec_proj=False,
    ):
        super(MTGNN, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=residual_channels, out_channels=residual_channels, kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)
        self.idx = torch.arange(self.num_nodes).to(device)


        # stuff for pretrain + finetune
        
        # some assertions
        assert self.gcn_true == True
        assert self.buildA_true == True

        self.mask_token = nn.Parameter(torch.randn(residual_channels), requires_grad=True)
        self.to_feat_embedding = nn.Linear(in_dim, residual_channels)
        
        # TODO: more complex structure decoder
        self.structure_decoder = InnerProductDecoder(stru_dec_drop, act=lambda x: x, with_proj=stru_dec_proj, hid_dim=skip_channels)

        # TODO: more complex feature decoder
        self.feature_decoder = nn.Conv2d(skip_channels, horizon, kernel_size=(1, 1), bias=True)


    def get_support(self):
        """
        [N, N] adjacency matrix
        """
        return self.gc(self.idx)


    def feature_masking(self, x, mask_ratio, mask_f_strategy='uniform', mask=None, *args, **kwargs):
        """
        masking according to strategy: per sample masking on time axis
        x: [B, Tin, N, Din]
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
        return [N, N] mask for mtgnn
        """
        B, _, N, _ = x.shape

        rw_percent = -1

        if mask is not None:
            return mask, rw_percent

        if mask_ratio != 0:
            pre_mask = kwargs.get('pre_mask', False)
            if not pre_mask:
                assert mask_s_strategy != 'uniform', "Please use <uniform_post_ce> strategy instead for MTGNN if post mask"
        else:
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
            binaried_support = (current_support > 0).to(current_support.dtype)

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
            binaried_support = (current_support > 0).to(current_support.dtype)
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

            # STEP1: random-walk based path masking: fixed graph 
            current_support = self.get_support()    # [N, N]
            binaried_support = (current_support > 0).to(current_support.dtype)

            num_edges = torch.count_nonzero(binaried_support).item()
            goal_discard = round(num_edges * mask_ratio)
            walks_per_node = kwargs.get('walks_per_node', 3)
            walk_length = kwargs.get('walk_length', 10)
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


    def encoding(self, input, s_mask, idx=None, pre_mask=False):
        """
        encoding masked input feature with masked supports
        input: [B, T, N, C]
        output: [B, F, N, 1]
        """
        input = rearrange(input, 'b t n c -> b c n t')
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        # get support and apply mask
        adp = self.gc(self.idx) * s_mask if not pre_mask else self.gc(self.idx, s_mask)

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
    
        return skip


    def encode(self, x, mask_s=0, mask_f=0, mask_s_strategy='uniform', mask_f_strategy='uniform', *args, **kwargs):
        """
        input: [B, Tin, N, Din]
        output: [B, F(t), N, 1]
        """
        B, T, N, _ = x.shape

        raw_x = x.clone()
        
        # random_masking for now: TODO: other strategies
        x_masked, f_mask = self.feature_masking(raw_x, mask_f, mask_f_strategy, mask=kwargs.get('f_mask', None), *args, **kwargs)

        # feature embedding with mask embedding
        embed_x = self.to_feat_embedding(x_masked)  # [B, T, N, F]
        embed_x = embed_x * f_mask.unsqueeze(-1) + repeat(self.mask_token, 'd -> b t n d', b=B, t=T, n=N) * (1 - f_mask.unsqueeze(-1))

        # structure mask: in gwnet, its a list
        s_mask, rw_percent = self.structure_masking(raw_x, mask_s, mask_s_strategy, mask=kwargs.get('s_mask', None), *args, **kwargs)

        # encode
        summary = self.encoding(embed_x, s_mask, idx=None, pre_mask=kwargs.get('pre_mask', False))

        return None, summary, f_mask, s_mask, rw_percent


    def decode_structure(self, summary):
        """
        summary: [B, F(t), N, 1]
        output: [B, N, N], between (0, 1)
        TODO: add inter projecter? Use node_embedding instead of summary?
        """
        summary = rearrange(summary, 'b t n f -> b n t f').squeeze(-1)
        return self.structure_decoder(summary)


    def decode_feature(self, summary):
        """
        summary: [B, F(t), N, 1]
        output: [B, Tin, N, 1]
        TODO: add inter projecter? Use embedding instead of summary?
        """
        return self.feature_decoder(summary)


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
            target = (target > 0).to(pred.dtype)
            target = repeat(target, 'm n -> b m n', b=B)
            si_mask = repeat(abs(s_mask - 1), 'm n -> b m n', b=B)
            if torch.sum(si_mask) != 0:
                diff = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * si_mask
                loss = torch.sum(diff) / torch.sum(si_mask)
            else:
                diff = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
                loss = torch.mean(diff)

        elif l_type == 'cls_boost':
            warnings.warn('Using cls_boost for structural loss for MTGNN can cause issues.')
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
        _, summary, f_mask, s_mask, rw_percent = self.encode(x, mask_s, mask_f, mask_s_strategy, mask_f_strategy, *args, **kwargs)

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


class MTGNN_Decoder(nn.Module):
    """
    agcrn decoder: exactly follow STGCL
    """
    def __init__(self, skip_channels, end_channels, horizon, de_mlp=True):
        super(MTGNN_Decoder, self).__init__()
        self.de_mlp = de_mlp
        assert self.de_mlp == True
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=horizon, kernel_size=(1,1), bias=True)
        
    def forward(self, input):
        """
        input: [B, C, N, 1]
        output: [B, Tout, N, 1]
        """
        x = F.relu(input)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

