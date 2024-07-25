import sys
sys.path.append('../')
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from .pf_utils import calculate_scaled_laplacian, calculate_random_walk_matrix
from einops import rearrange, repeat
from .pf_utils import mask_path
import warnings


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str, device=torch.device('cpu')):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self.device = device

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=self.device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=self.device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class DCGRUCell(torch.nn.Module):
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True, device=torch.device('cpu')
    ):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        self.filter_type = filter_type
        self.orig_adj_mx = adj_mx
        self.device = device

        self._fc_params = LayerParams(self, 'fc', self.device)
        self._gconv_params = LayerParams(self, 'gconv', self.device)

    def _get_all_supports(self, s_mask, normalize=False, filter_type='dual_random_walk'):
        """
        get all supports: assign self._supports attribute after structural masking
        """
        self._supports = []
        if s_mask is not None:
            adj_mx = self.apply_s_mask(self.orig_adj_mx, s_mask, normalize)
        else:
            adj_mx = self.orig_adj_mx
        supports = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
            supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))
        return 

    def _build_sparse_matrix(self, L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=self.device)
        return L

    def apply_s_mask(self, support, s_mask, normalize=False):
        # TODO: apply structural mask here?
        N, _ = support.shape
        
        # in dcrnn, support has to be not masked on diagnal
        support_masked = torch.from_numpy(support).to(self.device) * s_mask.fill_diagonal_(1)
        if normalize:
            support_masked = F.softmax(support_masked, dim=1)

        return support_masked.cpu()

    def forward(self, inputs, hx, s_mask=None, normalize=False):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        self._get_all_supports(s_mask, normalize, self.filter_type)
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        ## check sparse matrix 
        # print(self._supports[0]._nnz())

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class Seq2SeqAttrs:
    def __init__(
        self, 
        adj_mx,
        max_diffusion_step,
        cl_decay_steps,
        filter_type,
        num_nodes,
        num_rnn_layers,
        rnn_units,
    ):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(max_diffusion_step)
        self.cl_decay_steps = int(cl_decay_steps)
        self.filter_type = filter_type
        self.num_nodes = int(num_nodes)
        self.num_rnn_layers = int(num_rnn_layers)
        self.rnn_units = int(rnn_units)
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(
        self, 
        adj_mx,
        # Seq2SeqAttrs
        max_diffusion_step,
        cl_decay_steps,
        filter_type,
        num_nodes,
        num_rnn_layers,
        rnn_units,
        # Encoder
        input_dim,
        seq_len,
        device=torch.device('cpu'), 
    ):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, max_diffusion_step, cl_decay_steps, filter_type, num_nodes, num_rnn_layers, rnn_units)
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.device = device
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type, device=self.device) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None, s_mask=None, normalize=False):
        """
        inputs: [B, N * D_in]
        hidden_state: [L, B, N * F]
        return: output: (B, N * F)
                hidden_state: (L, B, N * F)
                (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], s_mask, normalize)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(
        self, 
        adj_mx,
        # Seq2SeqAttrs
        max_diffusion_step,
        cl_decay_steps,
        filter_type,
        num_nodes,
        num_rnn_layers,
        rnn_units,
        # Decoder
        output_dim,
        horizon,
        de_mlp=False,
        device=torch.device('cpu'),
    ):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, max_diffusion_step, cl_decay_steps, filter_type, num_nodes, num_rnn_layers, rnn_units)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)
        self.de_mlp = de_mlp
        if not self.de_mlp:
            self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        else:
            self.proj_1 = nn.Linear(self.rnn_units, self.rnn_units * 8)
            self.proj_2 = nn.Linear(self.rnn_units * 8, self.output_dim)
        self.device = device
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type, device=self.device) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None, s_mask=None, normalize=False):
        """
        Decoder forward pass.
        inputs: [B, N * D_out)
        hidden_state: [L, B, N * F]
        return: output: [B, N * F]
                hidden_state: [L, B, N * F]
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], s_mask, normalize)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        if not self.de_mlp:
            projected = self.projection_layer(output.view(-1, self.rnn_units))
        else:
            output = F.relu(self.proj_1(output.view(-1, self.rnn_units)))
            projected = self.proj_2(output)

        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


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


class DCRNN(nn.Module, Seq2SeqAttrs):
    def __init__(
        self, 
        adj_mx,
        # shared params
        num_nodes,
        num_rnn_layers,
        rnn_units,
        input_dim,
        seq_len,
        # dcrnn specific
        max_diffusion_step,
        cl_decay_steps,
        filter_type,
        # extras
        stru_dec_drop,
        stru_dec_proj,
        device=torch.device('cpu')
    ):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, max_diffusion_step, cl_decay_steps, filter_type, num_nodes, num_rnn_layers, rnn_units)

        self.mask_token = nn.Parameter(torch.randn(rnn_units), requires_grad=True)
        self.encoder = EncoderModel(adj_mx, max_diffusion_step, cl_decay_steps, filter_type, num_nodes, num_rnn_layers, rnn_units, rnn_units, seq_len, device)

        # TODO: more complex structure decoder
        self.structure_decoder = InnerProductDecoder(stru_dec_drop, act=lambda x: x, with_proj=stru_dec_proj, hid_dim=rnn_units)

        # TODO: more complex feature decoder
        self.feature_decoder = nn.Conv2d(1, seq_len, kernel_size=(1, rnn_units), bias=True)
        
        self.to_feat_embedding = nn.Linear(input_dim, rnn_units)

        self.cl_decay_steps = int(cl_decay_steps)
        self.device = device


    def encoding(self, inputs, s_mask):
        """
        input: [T_in, B, N * D_in]
        :return: [L, B, N * F]
        """
        inputs = rearrange(inputs, 'b t n f -> t b (n f)')
        encoder_hidden_state = None
        for t in range(self.encoder.seq_len):
            _, encoder_hidden_state = self.encoder(inputs[t], encoder_hidden_state, s_mask)

        encoder_hidden_state = rearrange(encoder_hidden_state, 'l b (n f) -> b l n f', n=self.num_nodes)
        return encoder_hidden_state


    def get_support(self):
        """
        get current support: [N, N]: between 0 to 1 element-wise
        NOTE: this is with self-loops
        """
        return torch.from_numpy(self.adj_mx).to(self.device)


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

        if mask_ratio != 0: 
            assert mask_s_strategy != 'uniform', "Please use <uniform_post_ce> strategy instead for DCRNN"
        else:
            return torch.ones(N, N, device=x.device), rw_percent

        if mask_s_strategy == 'uniform_post_ce':

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
                ne_edge_indices = torch.nonzero(binaried_support == 0)
                num_non_edges = ne_edge_indices.size(0)
                shuffled_idx = torch.randperm(num_non_edges)
                mask_indices = shuffled_idx[:num_discard]

                actual_mask_idx = ne_edge_indices[mask_indices]
                mask[actual_mask_idx[:, 0], actual_mask_idx[:, 1]] = 0

        elif mask_s_strategy == 'rw':
            
            with_negative = kwargs.get('with_negative', False)

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

            with_negative = kwargs.get('with_negative', False)
            if with_negative:
                # negative edge selection
                ne_edge_indices = torch.nonzero(abs(binaried_support - 1))
                num_non_edges = ne_edge_indices.size(0)
                shuffled_idx = torch.randperm(num_non_edges)
                mask_indices = shuffled_idx[:num_discard]
                
                actual_mask_idx = ne_edge_indices[mask_indices]
                mask[actual_mask_idx[:, 0], actual_mask_idx[:, 1]] = 0

        else:
            raise NotImplementedError

        return mask, rw_percent


    def encode(self, x, mask_s=0, mask_f=0, mask_s_strategy='uniform', mask_f_strategy='uniform', *args, **kwargs):
        """
        input: [B, Tin, N, Din]
        output: [B, Tin, N, F], [B, N, F]
        """
        B, T, N, _ = x.shape

        raw_x = x.clone()
        
        # random_masking for now: TODO: other strategies
        x_masked, f_mask = self.feature_masking(raw_x, mask_f, mask_f_strategy, mask=kwargs.get('f_mask', None), *args, **kwargs)

        # feature embedding with mask embedding
        embed_x = self.to_feat_embedding(x_masked)  # [B, T, N, F]
        embed_x = embed_x * f_mask.unsqueeze(-1) + repeat(self.mask_token, 'd -> b t n d', b=B, t=T, n=N) * (1 - f_mask.unsqueeze(-1))

        # structure mask
        s_mask, rw_percent = self.structure_masking(raw_x, mask_s, mask_s_strategy, mask=kwargs.get('s_mask', None), *args, **kwargs)
        
        # encode
        embedding = self.encoding(embed_x, s_mask)      # [B, L, N, F]
        summary = embedding[:, -1:, :, :].squeeze(1)    # [B, N, F]

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
            warnings.warn('Using cls_boost for structural loss for DCRNN can cause issues.')
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


    def forward(self, x, batches_seen=0, mask_s=0, mask_f=0, mask_s_strategy='uniform', mask_f_strategy='uniform', *args, **kwargs):
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

        if batches_seen == 0:
            print(">>> pretrainer params: {:.2f}M".format(sum(p.numel() for p in self.parameters() if p.requires_grad) / 1000000.0))

        return loss, loss_info


class DCRNN_Decoder(nn.Module):
    """
    dcrnn decoder: follow actural dcrnn decoder
    """
    def __init__(
        self, 
        adj_mx, 
        # shared
        num_nodes, 
        num_rnn_layers, 
        rnn_units, 
        output_dim,
        horizon, 
        # dcrnn specific
        max_diffusion_step, 
        cl_decay_steps, 
        filter_type, 
        # extras
        use_curriculum_learning, 
        de_mlp=False, 
        device=torch.device('cpu')
    ):
        super(DCRNN_Decoder, self).__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, max_diffusion_step, cl_decay_steps, filter_type, num_nodes, num_rnn_layers, rnn_units)

        self.de_mlp = de_mlp
        self.output_dim = output_dim
        self.horizon = horizon
        self.decoder = DecoderModel(adj_mx, max_diffusion_step, cl_decay_steps, filter_type, num_nodes, num_rnn_layers, rnn_units, output_dim, horizon, de_mlp, device)
        self.use_curriculum_learning = use_curriculum_learning
        self.device = device

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def decoding(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        encoder_hidden_state: [L, B, N * D_in]
        labels: [T_out, B, N * D_out] [optional, not exist for inference]
        batches_seen: global step [optional, not exist for inference]
        output: [T_out, B, N * D_out]
        """
        encoder_hidden_state = rearrange(encoder_hidden_state, 'b l n f -> l b (n f)')
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.output_dim),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.horizon):
            decoder_output, decoder_hidden_state = self.decoder(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        outputs = rearrange(outputs, 't b (n d) -> b t n d', n=self.num_nodes)
        return outputs

    def forward(self, x, labels=None, batches_seen=None):
        """
        input: [B, L, N, F], labels: [B, Tout, N, Cout]
        """
        if labels is not None:
            labels = rearrange(labels, 'b t n d -> t b (n d)')
        x = self.decoding(x, labels, batches_seen)

        if batches_seen == 0:
            print(">>> decoder params: {:.2f}M".format(sum(p.numel() for p in self.parameters() if p.requires_grad) / 1000000.0))
        return x


if __name__ == "__main__":
    B = 2
    Tin = 12
    N = 2
    D = 1
    Tout = 12


    torch.manual_seed(0)
    x = torch.randn(B, Tin, N, D)

    mdl= DCRNN(
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

    decoder = DCRNN_Decoder(out_dim=1, rnn_units=2, horizon=Tout, ft=True)
    _, encoded, _, _ = mdl.encode(x, mask_f_strategy='patch_uniform', mask_f=0.75, patch_length=3)
    exit()
    decoded = decoder(encoded.unsqueeze(1))
    print(encoded.shape)
    print(decoded.shape)