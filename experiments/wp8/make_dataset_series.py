import numpy as np
from pathlib import Path

from deepspace.utils.data import read_numpy, save_images, save_npy, to_uint8, normalization
from commontools.setup import config, logger


def get_paths():
    step = config.deepspace.step
    size = config.deepspace.size
    shape = config.deepspace.bin_shape
    paths = []
    if config.deepspace.data_format == 'npy':
        for index_x in range(0, shape[0] - size, step):
            for index_y in range(0, shape[1] - size, step):
                for index_z in range(0, shape[2] - size, step):
                    path = Path(config.deepspace.dataset_root) / ('_'.join([str(index_x), str(index_y), str(index_z)]) + '.' + config.deepspace.data_format)
                    paths.append(path)
    else:
        for index in range(shape[0]):
            path = Path(config.deepspace.dataset_root) / (str(index) + '.' + config.deepspace.data_format)
            paths.append(path)
    return paths


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
    # get paths to the target datas
    paths = get_paths()
    if config.deepspace.data_format == 'npy':
        dataset = []
        for index_x in range(0, shape[0] - size, step):
            for index_y in range(0, shape[1] - size, step):
                for index_z in range(0, shape[2] - size, step):
                    dataset.append(data[index_x:index_x+size, index_y:index_y+size, index_z:index_z+size])
        save_npy(dataset, paths, is_tqdm=True)
    else:
        if config.deepspace.data_format == 'png':
            data = to_uint8(data)
        save_images(data=data, paths=paths)


if __name__ == '__main__':
    make_dataset()
