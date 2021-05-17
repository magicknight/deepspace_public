"""
Defect images Data loader, for training on normal images
"""
from tokenize import generate_tokens
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as standard_transforms

from deepspace.augmentation.voxel import ToTensor, RandomBreak, RandomPatch, get_index, IndexPatch

from commontools.setup import config, logger


class NPYImages:
    def __init__(self, data_length=0, train_transform=None, target_transform=None):
        # get all the image paths
        self.train_data = np.load(config.deepspace.train_data, allow_pickle=True)
        self.target_data = np.load(config.deepspace.target_data, allow_pickle=True)
        self.data_length = data_length
        self.train_transform = train_transform
        self.target_transform = target_transform
        self.patcher = IndexPatch(size=config.deepspace.image_size)

    def __getitem__(self, index):
        """return png images

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        index = get_index(data_3d=self.train_data, size=config.deepspace.image_size)
        train_data = self.patcher(index, self.train_data)
        target_data = self.patcher(index, self.target_data)
        if self.train_transform is not None:
            train_data = self.train_transform(train_data)
        if self.target_transform is not None:
            target_data = self.target_transform(target_data)
        return train_data, target_data

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return self.data_length


class NPYTestImages:
    def __init__(self, transform=None):
        # get all the image paths
        self.size = config.deepspace.image_size
        self.data = np.load(config.deepspace.train_data, allow_pickle=True)
        self.data_set = self.data.unfold(0, *self.size).unfold(1, *self.size).unfold(2, *self.size).reshape(-1, *self.size)
        self.transform = transform

    def __getitem__(self, index):
        """return png images

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        data = self.data_set[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return self.data_set.shape[0]


class NPYDataLoader:
    def __init__(self):
        # transform
        self.train_trainsform = standard_transforms.Compose([
            ToTensor(),
            RandomBreak()
        ])
        self.target_transform = standard_transforms.Compose([
            ToTensor(),
        ])
        self.test_transform = standard_transforms.Compose([
            ToTensor(),
        ])
        # split dataset
        train_set = NPYImages(data_length=config.deepspace.train_data_length, train_transform=self.train_trainsform, target_transform=self.target_transform)
        valid_set = NPYImages(data_length=config.deepspace.validate_data_length, train_transform=self.train_trainsform, target_transform=self.target_transform)
        if config.deepspace.mode == 'train':
            # training needs train dataset and validate dataset
            self.train_loader = DataLoader(train_set, batch_size=config.deepspace.train_batch, shuffle=True,
                                           num_workers=config.deepspace.data_loader_workers)
            self.valid_loader = DataLoader(valid_set, batch_size=config.deepspace.validate_batch, shuffle=False,
                                           num_workers=config.deepspace.data_loader_workers)
            self.train_iterations = (len(train_set) + config.deepspace.train_batch) // config.deepspace.train_batch
            self.valid_iterations = (len(valid_set) + config.deepspace.validate_batch) // config.deepspace.validate_batch

        elif config.deepspace.mode == 'test':
            test_set = NPYTestImages(transform=self.test_transform)
            self.test_loader = DataLoader(test_set, batch_size=config.deepspace.test_batch, shuffle=False,
                                          num_workers=config.deepspace.data_loader_workers)
            self.test_iterations = (len(test_set) + config.deepspace.test_batch) // config.deepspace.test_batch

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
