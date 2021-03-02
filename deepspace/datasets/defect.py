"""
Defect images Data loader
"""
import imageio
import os
import torch
import numpy as np
import torchvision.utils as v_utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pathlib import Path

from deepspace.config import config, logger


def get_paths(mode):
    assert mode in ['train', 'val', 'validate', 'test', 'inference']
    root = Path(config.settings.dataset_root)
    items = []
    if mode == 'train':
        image_path = root / 'train' / 'good'
    elif mode == 'val' or mode == 'validate':
        image_path = root / 'train' / 'good'
    elif mode == 'test' or mode == 'inference':
        image_path = root / 'test'
        ground_truth_path = root / 'ground_truth'
    else:
        raise Exception("Please choose proper mode for data")
    items = list(image_path.glob('**/**.' + config.settings.data_format))
    if mode == 'test':
        ground_truth_items = list(ground_truth_path.glob('**/**.' + config.settings.data_format))
    else:
        ground_truth_items = None

    return items, ground_truth_items


class DefectImages:
    def __init__(self, mode, transform=None):
        self.imgs, self.masks = get_paths(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.imgs[index]
            image_data = imageio.imread(img_path)
            if self.transform is not None:
                image_data = self.transform(image_data)
            return image_data

        if self.mode == 'test':
            img_path = self.imgs[index]
            mask_path = self.masks[index]
            image = imageio.imread(img_path)
            mask = imageio.imread(mask_path)
            mask = mask.astype(dtype=np.uint8)

    def __len__(self):
        return len(self.imgs)


class DefectDataLoader:
    def __init__(self, config):
        assert config.mode in ['train', 'test', 'validate']

        self.input_transform = None

        if config.mode == 'train':
            train_set = DefectImages('train', config.data_root, transform=self.input_transform)
            valid_set = DefectImages('validate', config.data_root, transform=self.input_transform)

            self.train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                                           num_workers=config.data_loader_workers,
                                           pin_memory=config.pin_memory)
            self.valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False,
                                           num_workers=config.data_loader_workers,
                                           pin_memory=config.pin_memory)
            self.train_iterations = (len(train_set) + config.batch_size) // config.batch_size
            self.valid_iterations = (len(valid_set) + config.batch_size) // config.batch_size

        elif config.mode == 'test':
            test_set = DefectImages('test', config.data_root, transform=self.input_transform)

            self.test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False,
                                          num_workers=config.data_loader_workers,
                                          pin_memory=config.pin_memory)
            self.test_iterations = (len(test_set) + config.batch_size) // config.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
