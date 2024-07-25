from __future__ import absolute_import
import logging
import os
import torch
import pandas as pd
import numpy as np

def save_csv_log(cfg, head, value, is_create=False, file_property='log', file_name='sample'):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    if file_property == 'log':
        file_path = cfg.log_dir + '/{}.csv'.format(file_name)
    else:
        file_path = cfg.result_dir + '/{}.csv'.format(file_name)
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, 'a') as f:
            df.to_csv(f, header=False, index=False)


def save_ckpt(cfg, trainer, is_best=True, file_name=['sample_model_best.pth', 'sample_model_last.pth']):
    file_path = os.path.join(cfg.model_dir, file_name[1])
    trainer.save(file_path)
    if is_best:
        file_path = os.path.join(cfg.model_dir, file_name[0])
        trainer.save(file_path)