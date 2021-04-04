import numpy as np
from pathlib import Path

from deepspace.utils.data import read_numpy, save_images, to_uint8
from commontools.setup import config, logger


def make_dataset():
    data = read_numpy(config.deepspace.bin_file_path, shape=tuple(config.deepspace.bin_shape), datatype=getattr(np, config.deepspace.data_type))
    if config.deepspace.data_format == 'png':
        data = to_uint8(data)
    paths = []
    if config.deepspace.transpose:
        data = data.transpose(config.deepspace.transpose_order)
    for index in range(data.shape[0]):
        path = Path(config.deepspace.dataset_root) / (str(index) + '.' + config.deepspace.data_format)
        paths.append(path)
    save_images(data=data, paths=paths)


if __name__ == '__main__':
    make_dataset()
