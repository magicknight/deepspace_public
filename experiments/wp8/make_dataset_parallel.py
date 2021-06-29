'''
 ┌─────────────────────────────────────────────────────────────┐
 │┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
 ││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
 │├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
 ││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
 │├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
 ││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
 │├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
 ││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
 │└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
 │      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
 │      └───┴─────┴───────────────────────┴─────┴───┘          │
 └─────────────────────────────────────────────────────────────┘

Description: 
Author: Zhihua Liang
Github: https://github.com/magicknight
Date: 2021-06-28 19:35:34
LastEditors: Zhihua Liang
LastEditTime: 2021-06-28 19:35:34
FilePath: /home/zhihua/framework/deepspace/experiments/wp8/make_dataset_parallel.py
'''

import numpy as np
from pathlib import Path
from tqdm import tqdm

from deepspace.utils.data import read_bin, normalization
from commontools.setup import config, logger


def make_dataset():
    step = config.deepspace.step
    size = config.deepspace.size
    shape = config.deepspace.bin_shape
    # read data
    data = read_bin(config.deepspace.bin_file_path, shape=tuple(config.deepspace.bin_shape), datatype=getattr(np, config.deepspace.data_type))
    if 'target_data_type' in config.deepspace:
        data = data.astype(getattr(np, config.deepspace.target_data_type))
    if config.deepspace.transpose:
        data = data.transpose(config.deepspace.transpose_order)
    if config.deepspace.normalization:
        data = normalization(data)

    index_x = np.arange(0, shape[0] - size, step)
    index_y = np.arange(0, shape[1] - size, step)
    index_z = np.arange(0, shape[2] - size, step)
    x_idx, y_idx, z_idx = np.meshgrid(index_x, index_y, index_z)

    def save_data(args):
        x_idx, y_idx, z_idx = args
        image = data[x_idx:x_idx+size, y_idx:y_idx+size, z_idx:z_idx+size]
        path = Path(config.deepspace.dataset_root) / ('_'.join([str(x_idx), str(y_idx), str(z_idx)]) + '.' + config.deepspace.data_format)
        np.save(path, image)

    list(map(save_data, tqdm(zip(x_idx.flatten(), y_idx.flatten(), z_idx.flatten()), desc='wp8 dataset', total=len(x_idx.flatten()))))


if __name__ == '__main__':
    make_dataset()
