import numpy as np
from pathlib import Path
from tqdm import tqdm

from deepspace.utils.data import read_numpy, normalization
from commontools.setup import config, logger


def make_dataset():
    step = config.deepspace.step
    size = config.deepspace.size
    shape = config.deepspace.bin_shape
    # read data
    data = read_numpy(config.deepspace.bin_file_path, shape=tuple(config.deepspace.bin_shape), datatype=getattr(np, config.deepspace.data_type))
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
