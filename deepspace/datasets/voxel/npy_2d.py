"""
Defect images Data loader, for training on normal images
"""
from tokenize import generate_tokens
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms
from pprint import pprint
from deepspace.augmentation.image import CornerErasing, RandomErasing, RandomBreak
from commontools.setup import config, logger


class NPYImages:
    def __init__(self, train_file=None, target_file=None, train_transform=None, target_transform=None):
        # get all the image paths
        self.train_data = np.load(train_file, allow_pickle=True)
        self.target_data = np.load(target_file, allow_pickle=True)

        image_size = config.deepspace.image_size
        self.train_data = np.pad(self.train_data, [(0, 0), (0, image_size[0] - self.train_data.shape[1]), (0, image_size[1] - self.train_data.shape[2])])
        self.target_data = np.pad(self.target_data, [(0, 0), (0, image_size[0] - self.target_data.shape[1]), (0, image_size[1] - self.target_data.shape[2])])

        self.train_transform = train_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """return png images

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        train_data = self.train_data[index, :, :]
        target_data = self.target_data[index, :, :]
        if self.train_transform is not None:
            train_data = self.train_transform(train_data)
        if self.target_transform is not None:
            target_data = self.target_transform(target_data)

        return train_data, target_data

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return self.train_data.shape[0]


class NPYTestImages:
    def __init__(self, transform=None):
        self.transform = transform
        # process data first
        self.data = np.load(config.deepspace.train_data, allow_pickle=True)
        self.transform = transform

    def __getitem__(self, index):
        """return png images

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        data = self.data[index, :, :]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return self.data.shape[0]


class NPYDataLoader:
    def __init__(self):
        # transform
        self.train_trainsform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop(config.deepspace.crop_size),
            # transforms.Resize(config.deepspace.image_resolution),
            # CornerErasing(config.deepspace.image_resolution/2)
            # RandomBreak(
            #     probability=config.deepspace.break_probability,
            #     sl=config.deepspace.min_erasing_area,
            #     sh=config.deepspace.max_erasing_area,
            #     value=config.deepspace.value,
            #     return_normal=False,
            # )
        ])
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if config.deepspace.mode == 'train':
            train_set = NPYImages(train_file=config.deepspace.train_data, target_file=config.deepspace.target_data, train_transform=self.train_trainsform, target_transform=self.target_transform)
            valid_set = NPYImages(train_file=config.deepspace.valid_data, target_file=config.deepspace.valid_target_data, train_transform=self.train_trainsform, target_transform=self.target_transform)
            # training needs train dataset and validate dataset
            self.train_loader = DataLoader(train_set, batch_size=config.deepspace.train_batch, shuffle=True, num_workers=config.deepspace.data_loader_workers)
            self.valid_loader = DataLoader(valid_set, batch_size=config.deepspace.validate_batch, shuffle=True, num_workers=config.deepspace.data_loader_workers)
            self.train_iterations = (len(train_set) + config.deepspace.train_batch) // config.deepspace.train_batch
            self.valid_iterations = (len(valid_set) + config.deepspace.validate_batch) // config.deepspace.validate_batch

        elif config.deepspace.mode == 'test':
            test_set = NPYTestImages(train_file=config.deepspace.valid_data, target_file=config.deepspace.valid_target_data, transform=self.test_transform)
            self.test_loader = DataLoader(test_set, batch_size=config.deepspace.test_batch, shuffle=False, num_workers=config.deepspace.data_loader_workers)
            self.test_iterations = (len(test_set) + config.deepspace.test_batch) // config.deepspace.test_batch
            self.test_data = test_set.data

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
