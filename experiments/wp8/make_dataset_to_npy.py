import numpy as np
from pathlib import Path
from tqdm import tqdm

from deepspace.utils.data import read_bin, normalization
from commontools.setup import config, logger


def make_dataset():
    source_path = Path(config.deepspace.bin_file_path)
    # read data
    data = read_bin(source_path, shape=tuple(config.deepspace.bin_shape), datatype=getattr(np, config.deepspace.data_type))
    if 'target_data_type' in config.deepspace:
        data = data.astype(getattr(np, config.deepspace.target_data_type))
    if config.deepspace.transpose:
        data = data.transpose(config.deepspace.transpose_order)
    if config.deepspace.normalization:
        data = normalization(data)

    path = Path(config.deepspace.dataset_root) / (source_path.stem + '.' + config.deepspace.data_format)
    np.save(path, data)


if __name__ == '__main__':
    make_dataset()
