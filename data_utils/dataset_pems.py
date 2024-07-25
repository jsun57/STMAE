import os
import numpy as np
import pandas as pd

import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class CustomDataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, device=None, dtype=None):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = torch.from_numpy(xs).to(device).to(dtype)
        self.ys = torch.from_numpy(ys).to(device).to(dtype)


    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class CustomPretrainDataLoader(object):
    def __init__(self, xs, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs = self.xs[permutation]
        self.xs = xs

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                yield x_i
                self.current_ind += 1

        return _wrapper()


class PretrainDataset(Dataset):
    def __init__(
        self, dataset_name, traj_len, dtype='float32', *args, **kwargs
    ):
        self.dataset_name = dataset_name
        self.traj_len = traj_len
        self.dtype = np.float64 if dtype.lower() == 'float64' else np.float32

        self.data = self._prepare_data()
        self.segments = self._generate_segments()

    def _prepare_data(self):
        data = np.load(os.path.join('./data/', self.dataset_name, 'traj.npz'))['x']
        return data

    def _generate_segments(self):
        segments = [(init, init + self.traj_len) for init in range(0, self.data.shape[0] - self.traj_len + 1)]
        return segments

    def _get_segments(self, init, end):
        traj = self.data[init:end,...]
        return traj

    def __getitem__(self, index):
        """
        x: normalized
        y: raw
        """
        (init, end) = self.segments[index]
        traj = self._get_segments(init, end)

        return traj

    def __len__(self):
        return len(self.segments)


def generate_pretrain_data(dataset_name, traj_len):
    """
    return: [all, traj_len, N, C]
    """
    raw_data = np.load(os.path.join('./data/', dataset_name, 'traj.npz'))['x']
    segments = np.array([(init, init + traj_len) for init in range(0, raw_data.shape[0] - traj_len + 1)])
    start_indices = segments[:, 0]
    end_indices = segments[:, 1]
    all_length = len(start_indices)
    T = np.max(end_indices - start_indices)
    segment_indices = np.arange(T)[None, :] + start_indices[:, None]
    new_data = raw_data[segment_indices]
    return new_data


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, device=None, dtype=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = CustomDataLoader(data['x_train'], data['y_train'], batch_size, device=device, dtype=dtype)
    data['val_loader'] = CustomDataLoader(data['x_val'], data['y_val'], valid_batch_size, device=device, dtype=dtype)
    data['test_loader'] = CustomDataLoader(data['x_test'], data['y_test'], test_batch_size, device=device, dtype=dtype)
    data['scaler'] = scaler
    return data


def load_pretrain_dataset(dataset_name, traj_len, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    data['x_train'] = generate_pretrain_data(dataset_name, traj_len)
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = CustomPretrainDataLoader(data['x_train'], batch_size)
    data['scaler'] = scaler
    return data
