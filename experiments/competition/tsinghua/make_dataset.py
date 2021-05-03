import numpy as np
from pathlib import Path
from tqdm import tqdm
import tables
import h5py

from deepspace.utils.data import save_npy
from commontools.setup import config, logger


def get_paths():
    root = Path(config.deepspace.dataset_root)
    first_round_root = root / 'first_round'
    target_root = root / 'npy' / 'first_round'
    geo_file = root / 'playground' / 'geo.h5'
    paths = first_round_root.glob('**/pre-*.' + config.deepspace.raw_data_format)
    return paths, target_root, geo_file


def save_data(args):
    file_id, event_id, channel_id, pe_time, vis = args
    paths, target_root, geo_file = get_paths()
    available_data = np.stack((channel_id, pe_time), axis=-1)
    available_data = np.pad(available_data, ((0, config.deepspace.data_size - available_data.shape[0]), (0, 0)), mode='constant', constant_values=0)
    filename = '_'.join([str(file_id), str(event_id), str(vis)]) + '.' + config.deepspace.data_format
    file_path = target_root / filename
    np.save(file_path, available_data)


def make_dataset():
    paths, target_root, geo_file = get_paths()

    # geometry
    # geo_table = tables.open_file(geo_file)
    # geo_dict = dict()
    # for index in range(len(geo_table.root.Geometry)):
    #     geo_dict[geo_table.root.Geometry[index][0]] = index
    # go through all h5 files
    for each_file in tqdm(paths, desc='files'):
        file_index = each_file.name.split('-')[-1].split('.')[0]
        h5file = h5py.File(each_file, 'r', libver='latest', swmr=True)
        petruth = h5file['PETruth']
        patruth = h5file['ParticleTruth']
        # watruth = h5file['Waveform']
        event_id_petruth, index_petruth_start = np.unique(petruth['EventID'], return_index=True)
        # index_petruth_end = np.append(index_petruth_start, len(petruth))[1:]
        # index_petruth = np.stack((index_petruth_start, index_petruth_end), axis=-1)

        # data = np.zeros((index_petruth.shape[0], config.deepspace.data_size, 2), dtype=np.float32)
        # data = []
        # target_paths = []
        vis = patruth['vis'].tolist()
        event_ids = petruth['EventID'].tolist()
        file_ids = [file_index] * len(event_ids)
        channel_ids = np.split(petruth['ChannelID'], index_petruth_start[1:])
        petimes = np.split(petruth['PETime'], index_petruth_start[1:])
        list(map(save_data, zip(file_ids, event_ids, channel_ids, petimes, vis)))
        # index = 0
        # for channel_id, petime in tqdm(zip(channel_ids, petimes), total=len(channel_ids)):
        #     event_id = petruth['EventID'][index]
        #     index += 1
        #     vis = patruth['vis'][event_id]
        #     available_data = np.stack((channel_id, petime), axis=-1)
        #     available_data = np.pad(available_data, ((0, config.deepspace.data_size - available_data.shape[0]), (0, 0)), mode='constant', constant_values=0)
        #     filename = '_'.join([str(file_index), str(event_id), str(vis)]) + '.' + config.deepspace.data_format
        #     file_path = target_root / filename
        #     data.append(available_data)
        #     target_paths.append(file_path)
        # save_npy(data, target_paths)
        # for pe_index in tqdm(index_petruth, desc='pe time data generating'):
        #     event_id = petruth['EventID'][pe_index[0]]
        #     vis = patruth['vis'][event_id]
        #     # channel_ids = petruth['ChannelID'][pe_index[0]:pe_index[1]]
        #     # petimes = petruth['PETime'][pe_index[0]:pe_index[1]]
        #     available_data = np.stack((channel_ids, petimes), axis=-1)
        #     available_data = np.pad(available_data, ((0, config.deepspace.data_size - available_data.shape[0]), (0, 0)), mode='constant', constant_values=0)
        #     # read data
        #     # data[0:len(channel_ids)] = available_data
        #     # data.append(np.pad(available_data, (0, config.deepspace.data_size - available_data.shape[0]), mode='constant', constant_values=0))
        #     filename = '_'.join([str(file_index), str(event_id), str(vis)]) + '.' + config.deepspace.data_format
        #     file_path = target_root / filename
        #     # target_paths.append(file_path)
        #     np.save(file_path, available_data)
        # # save_npy(data, target_paths)


if __name__ == '__main__':
    make_dataset()
