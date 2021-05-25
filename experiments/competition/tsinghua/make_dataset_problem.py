from pprint import pprint
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tables
import h5py

from deepspace.utils.data import normalization, save_npy
from commontools.setup import config, logger


def norm_waveform(data):
    min_value, max_value = tuple(config.deepspace.waveform_range)
    data = (max_value - data) / (max_value - min_value)
    return data.astype(np.float32)


def save_data(args):
    event_id, channel_ids, waveforms = args
    data = dict()

    def fill_data(args):
        index, waveform = args
        data[index] = norm_waveform(waveform)

    list(map(fill_data, zip(channel_ids, waveforms)))
    filename = str(event_id) + '.' + config.deepspace.data_format
    file_path = target_root / filename
    np.save(file_path, data)


def make_dataset():
    # go through all h5 files
    for each_file in tqdm(paths, desc='files'):
        h5file = h5py.File(each_file, 'r', libver='latest', swmr=True)
        watruth = h5file['Waveform']
        event_id_watruth, index_watruth_start = np.unique(watruth['EventID'], return_index=True)
        channel_ids = np.split(watruth['ChannelID'], index_watruth_start[1:])
        waveforms = np.split(watruth['Waveform'], index_watruth_start[1:])

        list(map(save_data, tqdm(zip(event_id_watruth, channel_ids, waveforms), total=len(event_id_watruth))))


if __name__ == '__main__':
    # get all paths
    root = Path(config.deepspace.dataset_root)
    target_root = root.parent / 'npy' / root.name
    geo_file = root / 'geo.h5'
    paths = list(root.glob(config.deepspace.data_patern + config.deepspace.raw_data_format))
    # geometry
    geo_info = h5py.File(geo_file, 'r', libver='latest', swmr=True)
    size_y = geo_info['Geometry']['ChannelID'].shape[0]
    detector_index = {k: v for v, k in enumerate(geo_info['Geometry']['ChannelID'])}
    logger.info(root)
    logger.info(target_root)
    logger.info(geo_file)
    logger.info(paths)
    make_dataset()
