from ssl import ALERT_DESCRIPTION_ILLEGAL_PARAMETER
import numpy as np
from pathlib import Path
from tqdm import tqdm
from imageio import imread
from deepspace.utils.data import read_numpy, normalization
from commontools.setup import config, logger


def make_npy(path):
    image_files = list(path.glob('**/*' + '.' + config.deepspace.source_data_format))
    data = list(map(imread, tqdm(image_files, desc='read data')))
    data = np.stack(data, axis=0)
    dataset_file = Path(config.deepspace.npy_root) / (path.name + '.' + config.deepspace.target_data_format)
    np.save(dataset_file, data)


def make_dataset():
    # get folders
    root = Path(config.deepspace.dataset_root)
    dataset_paths = list(root.glob('*'))
    logger.info(dataset_paths)
    list(map(make_npy, tqdm(dataset_paths, desc='bone dataset', )))


if __name__ == '__main__':
    make_dataset()
