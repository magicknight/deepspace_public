"""
Defect images Data loader, for training on normal images
"""
from tokenize import generate_tokens
import numpy as np
from numpy.lib.arraysetops import isin
import math
from einops import rearrange

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import torchvision.transforms as standard_transforms

from deepspace.augmentation.voxel import ToTensor, get_index, IndexPatch, Rotate
from commontools.setup import config, logger


class TrainDataset(object):
    def __init__(self, train_data, target_data, data_length=0, train_transform=None, target_transform=None):
        self.train_data = train_data
        self.target_data = target_data
        self.data_length = data_length
        self.train_transform = train_transform
        self.target_transform = target_transform
        # a tool to extract smaller 3D volumns from raw data
        self.patcher = IndexPatch(size=config.deepspace.image_size)
        self.rotate = Rotate()

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
            target_data = np.copy(self.patcher(index, self.target_data))
        # target data is non-break train data
        if self.target_transform is not None:
            target_data = self.target_transform(target_data)
        if self.train_transform is not None:
            train_data = self.train_transform(train_data)
        # # perform rotation
        steps = torch.randint(0, 4, (3,), device=train_data.device)
        train_data = self.rotate(train_data, steps=steps)
        target_data = self.rotate(target_data, steps=steps)

        return train_data, target_data

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return self.data_length


class ValiDataset(object):
    """
    Load the defect images, and return the data loader
    """

    def __init__(self, train_data, target_data, train_transform=None, target_transform=None):
        # get all the image paths
        self.train_transform = train_transform
        self.target_transform = target_transform

        real_index = np.random.randint(0, train_data.shape[0] - config.deepspace.image_size[0])
        self.train_data = np.copy(train_data[real_index : real_index + config.deepspace.image_size[0], :, :], order="C")
        self.target_data = np.copy(target_data[real_index : real_index + config.deepspace.image_size[0], :, :], order="C")
        # rearrange the data to fit the model
        with torch.no_grad():
            self.train_data = rearrange(self.train_data, "b1 (n1 b2) (n2 b3) -> (n1 n2) b1 b2 b3", b2=config.deepspace.image_size[1], b3=config.deepspace.image_size[2])
            self.target_data = rearrange(self.target_data, "b1 (n1 b2) (n2 b3) -> (n1 n2) b1 b2 b3", b2=config.deepspace.image_size[1], b3=config.deepspace.image_size[2])

    def __getitem__(self, index):
        """return png images

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        train_data = self.train_data[index]
        target_data = self.target_data[index]
        if self.train_transform is not None:
            train_data = self.train_transform(train_data)
        if self.target_transform is not None:
            target_data = self.target_transform(target_data)
        return train_data, target_data

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return len(self.train_data)


class TestDataset(object):
    """
    Load the defect images, and return the data loader
    """

    def __init__(self, transform=None, shared_array=None):
        # get all the image paths
        if shared_array is None:
            self.input_data = np.load(config.deepspace.test_dataset, allow_pickle=True)
        # take just a part of the input data if image_range set in config
        print("input data shape", self.input_data.shape)
        if "image_range" in config.deepspace:
            self.input_data = self.input_data[config.deepspace.image_range[0] : config.deepspace.image_range[1], :, :]
            print("input data shape", self.input_data.shape)
        # patch the input data to size of integer times of patch_size
        self.input_data = np.pad(self.input_data, ((0, math.ceil(self.input_data.shape[0] / config.deepspace.image_size[0]) * config.deepspace.image_size[0] - self.input_data.shape[0]), (0, 0), (0, 0)), "constant", constant_values=0)
        self.input_shape = self.input_data.shape
        self.input_transform = transform
        # 分解成多个patch
        with torch.no_grad():
            self.input_data = rearrange(self.input_data, "(n1 b1) (n2 b2) (n3 b3) -> (n1 n2 n3) b1 b2 b3", b1=config.deepspace.image_size[0], b2=config.deepspace.image_size[1], b3=config.deepspace.image_size[2])

    def __getitem__(self, index):
        """return png images

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        input_data = self.input_data[index, :, :]
        if self.input_transform is not None:
            input_data = self.input_transform(input_data)

        return input_data, index

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return self.input_data.shape[0]


class Loader:
    def __init__(self, shared_array=None):
        if isinstance(shared_array, list):
            self.train_data = shared_array[0]
            self.target_data = shared_array[1]
        if shared_array is None:
            self.train_data = np.load(config.deepspace.train_dataset, allow_pickle=True)
            self.target_data = np.load(config.deepspace.target_dataset, allow_pickle=True)
        # transform
        self.train_trainsform = standard_transforms.Compose(
            [
                ToTensor(),
            ]
        )
        self.target_transform = standard_transforms.Compose(
            [
                ToTensor(),
            ]
        )
        self.test_transform = standard_transforms.Compose(
            [
                ToTensor(),
            ]
        )
        if config.deepspace.mode == "train":
            # dataset
            train_set = TrainDataset(train_data=self.train_data, target_data=self.target_data, data_length=config.deepspace.train_data_length, train_transform=self.train_trainsform, target_transform=self.target_transform)
            valid_set = ValiDataset(train_data=self.train_data, target_data=self.target_data, train_transform=self.train_trainsform, target_transform=self.target_transform)
            # sampler
            if config.common.distribute:
                from torch_xla.core import xla_model

                train_sampler = DistributedSampler(
                    train_set,
                    num_replicas=xla_model.xrt_world_size(),
                    rank=xla_model.get_ordinal(),
                    shuffle=config.deepspace.shuffle,
                )
                validate_sampler = DistributedSampler(
                    valid_set,
                    num_replicas=xla_model.xrt_world_size(),
                    rank=xla_model.get_ordinal(),
                    shuffle=config.deepspace.shuffle,
                )
            else:
                train_sampler = None
                validate_sampler = None
            # training needs train dataset and validate dataset
            self.train_loader = DataLoader(train_set, batch_size=config.deepspace.train_batch, sampler=train_sampler, drop_last=config.deepspace.drop_last, num_workers=config.deepspace.data_loader_workers)
            self.valid_loader = DataLoader(valid_set, batch_size=config.deepspace.validate_batch, sampler=validate_sampler, drop_last=config.deepspace.drop_last, num_workers=config.deepspace.data_loader_workers)
            # self.valid_loader = self.train_loader
            self.train_iterations = len(train_set) // config.deepspace.train_batch
            self.valid_iterations = len(valid_set) // config.deepspace.validate_batch

        elif config.deepspace.mode == "test":
            test_set = TestDataset(transform=self.test_transform, shared_array=None)
            self.test_loader = DataLoader(test_set, batch_size=config.deepspace.test_batch, shuffle=False, num_workers=config.deepspace.data_loader_workers)
            self.test_iterations = len(test_set) // config.deepspace.test_batch
            self.input_shape = test_set.input_shape
            # self.test_data = test_set.data
            # self.data_shape = test_set.data_shape

        else:
            raise Exception("Please choose a proper mode for data loading")

    def finalize(self):
        pass
