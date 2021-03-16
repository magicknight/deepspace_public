import numpy as np
from pathlib import Path

from deepspace.utils.data import read_numpy, save_images, to_uint8
from deepspace.config.config import config, logger


def make_dataset():
    data = read_numpy(config.settings.bin_file_path, shape=tuple(config.settings.bin_shape), datatype=getattr(np, config.settings.data_type))
    if config.settings.data_format == 'png':
        data = to_uint8(data)
    paths = []
    if config.settings.transpose:
        data = data.transpose(config.settings.transpose_order)
    for index in range(data.shape[0]):
        path = Path(config.settings.dataset_root) / (str(index) + '.' + config.settings.data_format)
        paths.append(path)
    save_images(data=data, paths=paths)


if __name__ == '__main__':
    make_dataset()
