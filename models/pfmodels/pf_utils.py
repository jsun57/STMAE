import numpy as np
import torch
import torch.nn as nn
import torch_cluster
random_walk = torch.ops.torch_cluster.random_walk
from torch_geometric.utils import degree
import scipy.sparse as sp
from scipy.sparse import linalg
import warnings


def mask_path(
    binaried_support,
    mask_ratio, 
    walks_per_node=1,
    walk_length=3, 
    start='edge',
    p = 1.0,
    q = 1.0,
):

    assert start in ['node', 'edge']

    edge_index = binaried_support.nonzero().t().contiguous()
    num_edges = edge_index.size(1)

    edge_mask = edge_index.new_ones(num_edges, dtype=torch.bool)    # [num_edges]: True is keep, False is mask

    if mask_ratio == 0.0:
        return None, 0

    num_nodes = binaried_support.size(0)
    row, col = edge_index

    if start == 'edge':
        sample_mask = torch.rand(row.size(0), device=edge_index.device) <= mask_ratio
        start = row[sample_mask].repeat(walks_per_node)
    else:
        start = torch.randperm(num_nodes, device=edge_index.device)[:round(num_nodes * mask_ratio)].repeat(walks_per_node)

    deg = degree(row, num_nodes=num_nodes)

    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])
    n_id, e_id = random_walk(rowptr, col, start, walk_length, p, q)

    e_id = e_id[e_id != -1].view(-1)  # filter illegal edges
    edge_mask[e_id] = False

    remain_edge_index = edge_index[:, edge_mask]
    masked_edge_index = edge_index[:, ~edge_mask]   # [2, num_masked]

    num_masked = masked_edge_index.size(1)

    return masked_edge_index, num_masked


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)
