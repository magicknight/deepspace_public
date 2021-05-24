from pprint import pprint
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tables
import h5py

from deepspace.utils.data import save_npy
from commontools.setup import config, logger


def save_data(args):
    file_id, event_id, momentum, channel_ids, waveforms = args
    # data = np.zeros(config.deepspace.size)
    data = dict()

    def fill_data(args):
        index, waveform = args
        # data[detector_index[index]] = waveform
        data[index] = waveform

    list(map(fill_data, zip(channel_ids, waveforms)))
    filename = '_'.join([str(file_id), str(event_id), str(momentum)]) + '.' + config.deepspace.data_format
    file_path = target_root / filename
    np.save(file_path, data)


def make_dataset():
    # go through all h5 files
    for each_file in tqdm(paths, desc='files'):
        file_index = each_file.name.split('-')[-1].split('.')[0]
        h5file = h5py.File(each_file, 'r', libver='latest', swmr=True)

        # petruth = h5file['PETruth']
        patruth = h5file['ParticleTruth']
        watruth = h5file['Waveform']

        event_id_watruth, index_watruth_start = np.unique(watruth['EventID'], return_index=True)
        # index_watruth_end = np.append(index_watruth_start, len(watruth))[1:]
        # index_watruth = np.stack((index_watruth_start, index_watruth_end), axis=-1)

        channel_ids = np.split(watruth['ChannelID'], index_watruth_start[1:])
        waveforms = np.split(watruth['Waveform'], index_watruth_start[1:])

        momentum = patruth['p'].tolist()
        file_ids = [file_index] * len(event_id_watruth)

        list(map(save_data, tqdm(zip(file_ids, event_id_watruth, momentum, channel_ids, waveforms), total=len(event_id_watruth))))


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
