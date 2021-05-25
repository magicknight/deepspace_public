"""
Defect images Data loader, for training on normal images
"""
from tokenize import generate_tokens
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms

from commontools.setup import config, logger


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        data = torch.from_numpy(data)
        return data


def get_paths():
    """return paths of data
    Returns:
        list: items list, and masks list if in test mode
    """
    root = Path(config.deepspace.dataset_root)
    # return path
    paths = list(root.glob('**/*.' + config.deepspace.data_format))
    return paths


class NPYDataSet:
    def __init__(self, transform=None):
        # get all the image paths
        self.paths = get_paths()
        self.transform = transform

    def __getitem__(self, index):
        """return npy data
        Args:
            index (int): data index
        Returns:
            Tensor: images
        """
        # get data path
        path = self.paths[index]
        momentum = np.array(float(path.name.split('_')[-1].split('.'+config.deepspace.data_format)[0]), dtype=np.float64)
        # read in data. data shape: [2856, 1000], index shape: [2856]
        data = np.load(path, allow_pickle=True)
        # position 0 will be use for addition data, so all index move to the next position
        data_keys = np.array(list(data.item().keys())) + 1
        data_values = np.array(list(data.item().values()))
        # pad to avoid torch error on non equal size
        real_shape = np.array(data_keys.shape[0], dtype=np.int)
        data_keys = np.pad(data_keys, ((0, config.deepspace.pad_size - real_shape)), 'constant', constant_values=0)
        data_values = np.pad(data_values, ((0, config.deepspace.pad_size - real_shape), (0, 0)), 'constant', constant_values=0)

        if self.transform is not None:
            data_keys = self.transform(data_keys)
            data_values = self.transform(data_values)
            momentum = self.transform(momentum)
        return data_keys, data_values, momentum

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return len(self.paths)


class NPYTestDataSet:
    def __init__(self, transform=None):
        # get all the image paths
        self.paths = get_paths()
        self.transform = transform

    def __getitem__(self, index):
        """return npy data
        Args:
            index (int): data index
        Returns:
            Tensor: images
        """
        # get data path
        path = self.paths[index]
        event_id = int(path.name.split('.')[0])
        # read in data. data shape: [2856, 1000], index shape: [2856]
        data = np.load(path, allow_pickle=True)
        data_keys = np.array(list(data.item().keys())) + 1
        data_values = np.array(list(data.item().values()))
        # pad to avoid torch error on non equal size
        real_shape = np.array(data_keys.shape[0], dtype=np.int)
        data_keys = np.pad(data_keys, ((0, config.deepspace.pad_size - real_shape)), 'constant', constant_values=0)
        data_values = np.pad(data_values, ((0, config.deepspace.pad_size - real_shape), (0, 0)), 'constant', constant_values=0)

        if self.transform is not None:
            data_keys = self.transform(data_keys)
            data_values = self.transform(data_values)
        return data_keys, data_values, event_id

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return len(self.paths)


class NPYDataLoader:
    def __init__(self):
        # transform
        self.input_transform = transforms.Compose([
            ToTensor(),
        ])
        if config.deepspace.mode == 'train':
            # split dataset
            data_set = NPYDataSet(transform=self.input_transform)
            train_size = int(config.deepspace.ratio * len(data_set))
            valid_size = len(data_set) - train_size
            train_set, valid_set = torch.utils.data.random_split(data_set, [train_size, valid_size], generator=torch.Generator().manual_seed(config.deepspace.seed))
            # training needs train dataset and validate dataset
            self.train_loader = DataLoader(train_set, batch_size=config.deepspace.train_batch, shuffle=True, num_workers=config.deepspace.data_loader_workers)
            self.valid_loader = DataLoader(valid_set, batch_size=config.deepspace.validate_batch, shuffle=False, num_workers=config.deepspace.data_loader_workers)
            self.train_iterations = (len(train_set) + config.deepspace.train_batch) // config.deepspace.train_batch
            self.valid_iterations = (len(valid_set) + config.deepspace.validate_batch) // config.deepspace.validate_batch

        elif config.deepspace.mode == 'test':
            data_set = NPYTestDataSet(transform=self.input_transform)
            self.test_loader = DataLoader(data_set, batch_size=config.deepspace.test_batch, shuffle=False, num_workers=config.deepspace.data_loader_workers)
            self.test_iterations = (len(data_set) + config.deepspace.test_batch) // config.deepspace.test_batch

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
