"""
Defect images Data loader, for training on normal images
"""
from tokenize import generate_tokens
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import torchvision.transforms as standard_transforms

from deepspace.augmentation.voxel import ToTensor, RandomBreak, RandomPatch, get_index, IndexPatch
from commontools.setup import config, logger


class NPYImages:
    def __init__(self, data, data_length=0, train_transform=None, target_transform=None):
        self.train_data = data
        self.data_length = data_length
        self.train_transform = train_transform
        self.target_transform = target_transform
        # a tool to extract smaller 3D volumns from raw data
        self.patcher = IndexPatch(size=config.deepspace.image_size)

    def __getitem__(self, index):
        """return png images

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        index = get_index(data_3d=self.train_data, size=config.deepspace.image_size)
        with torch.no_grad():
            train_data = np.copy(self.patcher(index, self.train_data))
        # target data is non-break train data
        if self.target_transform is not None:
            target_data = self.target_transform(train_data)
        if self.train_transform is not None:
            train_data = self.train_transform(train_data)
        return train_data, target_data

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return self.data_length


class NPYTestImages:
    def __init__(self, data, transform=None):
        # get all the image paths
        self.size = config.deepspace.image_size
        if isinstance(self.size, (int)):
            self.size = [self.size] * 3
        self.transform = transform

        data = np.pad(data, [(0, size - side % size) for size, side in zip(self.size, data.shape)])
        self.data = self.transform(data)
        coordinates = []
        for index in range(3):
            coordinate = np.linspace(0, data.shape[index], int(data.shape[index]/config.deepspace.image_size[index]) + 1)
            coordinates.append(coordinate[0:-1])
        coordinates = np.meshgrid(coordinates[0], coordinates[1], coordinates[2])
        self.coordinates = np.stack(map(np.ravel, coordinates), axis=1)
        self.coordinates = self.coordinates.astype(np.int)
        # self.data = self.data.unfold(0, *self.size).unfold(1, *self.size).unfold(2, *self.size).reshape(-1, *self.size)

    def __getitem__(self, index):
        """return png images

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        coordinate = self.coordinates[index]
        data = self.data[:, coordinate[0]:coordinate[0]+self.size[0], coordinate[1]:coordinate[1]+self.size[1], coordinate[2]:coordinate[2]+self.size[2]]
        return data, coordinate

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return len(self.coordinates)


class Loader:
    def __init__(self, shared_array=None):
        # process data first
        # self.data = np.load(config.deepspace.train_dataset, allow_pickle=True)
        self.data = shared_array
        # transform
        self.train_trainsform = standard_transforms.Compose([
            ToTensor(),
            RandomBreak(probability=config.deepspace.break_probability, sides_range=config.deepspace.break_range)
        ])
        self.target_transform = standard_transforms.Compose([
            ToTensor(),
        ])
        self.test_transform = standard_transforms.Compose([
            ToTensor(),
        ])
        # dataset
        train_set = NPYImages(data=self.data, data_length=config.deepspace.train_data_length, train_transform=self.train_trainsform, target_transform=self.target_transform)
        valid_set = NPYImages(data=self.data, data_length=config.deepspace.validate_data_length, train_transform=self.train_trainsform, target_transform=self.target_transform)
        # sampler
        if config.common.distribute:
            from torch_xla.core import xla_model
            train_sampler = DistributedSampler(
                train_set,
                num_replicas=xla_model.xrt_world_size(),
                rank=xla_model.get_ordinal(),
                shuffle=False)
            validate_sampler = DistributedSampler(
                valid_set,
                num_replicas=xla_model.xrt_world_size(),
                rank=xla_model.get_ordinal(),
                shuffle=False)
        else:
            train_sampler = None
            validate_sampler = None
        if config.deepspace.mode == 'train':
            # training needs train dataset and validate dataset
            self.train_loader = DataLoader(
                train_set,
                batch_size=config.deepspace.train_batch,
                sampler=train_sampler,
                drop_last=config.deepspace.drop_last,
                shuffle=config.deepspace.shuffle,
                num_workers=config.deepspace.data_loader_workers
            )
            self.valid_loader = DataLoader(
                valid_set,
                batch_size=config.deepspace.validate_batch,
                sampler=validate_sampler,
                drop_last=config.deepspace.drop_last,
                shuffle=config.deepspace.shuffle,
                num_workers=config.deepspace.data_loader_workers
            )
            # self.valid_loader = self.train_loader
            self.train_iterations = (len(train_set) + config.deepspace.train_batch) // config.deepspace.train_batch
            self.valid_iterations = (len(valid_set) + config.deepspace.validate_batch) // config.deepspace.validate_batch

        elif config.deepspace.mode == 'test':
            test_set = NPYTestImages(transform=self.test_transform)
            self.test_loader = DataLoader(test_set, batch_size=config.deepspace.test_batch, shuffle=False, num_workers=config.deepspace.data_loader_workers)
            self.test_iterations = (len(test_set) + config.deepspace.test_batch) // config.deepspace.test_batch
            self.test_data = test_set.data

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
