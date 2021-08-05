"""
Defect images Data loader, for training on normal images
"""
from tokenize import generate_tokens
import numpy as np
from numpy.lib.arraysetops import isin
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import torchvision.transforms as standard_transforms
from einops import rearrange, reduce, repeat

from deepspace.augmentation.voxel import Rotate, ToTensor, RandomBreak, RandomPatch, get_index, IndexPatch, RandomRotate
from commontools.setup import config, logger


class NPYImages:
    def __init__(self, train_data, target_data, data_length=0, train_transform=None, target_transform=None):
        self.train_data = train_data
        self.target_data = target_data
        self.data_length = data_length
        self.train_transform = train_transform
        self.target_transform = target_transform
        # a tool to extract smaller 3D volumns from raw data
        self.patcher = IndexPatch(size=config.deepspace.image_size)
        self.rotate = Rotate()

    def __get_one_item__(self):
        """return npy images
        """
        index = get_index(data_3d=self.train_data, size=config.deepspace.image_size)
        with torch.no_grad():
            train_data = np.copy(self.patcher(index, self.train_data))
            target_data = np.copy(self.patcher(index, self.target_data))
        # target data is non-break train data
        if self.train_transform is not None:
            train_data = self.target_transform(train_data)
        if self.target_transform is not None:
            target_data = self.target_transform(target_data)
        return train_data, target_data

    def __getitem__(self, index):
        """return npy images

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        train_data, target_data = self.__get_one_item__()
        rot_1 = torch.randint(0, 4, (3, ), device=train_data.device)
        train_data_1 = self.rotate(train_data, rot_1)
        target_data_1 = self.rotate(target_data, rot_1)
        rot_2 = torch.randint(0, 4, (3, ), device=train_data.device)
        train_data_2 = self.rotate(train_data, rot_2)
        target_data_2 = self.rotate(target_data, rot_2)
        return train_data_1, target_data_1, rot_1, train_data_2, target_data_2, rot_2

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return self.data_length


class NPYTestImages:
    def __init__(self, transform=None, shared_array=None):
        # get all the image paths
        self.size = config.deepspace.image_size
        if isinstance(self.size, (int)):
            self.size = [self.size] * 3
        self.transform = transform
        if shared_array is None:
            self.train_data = np.load(config.deepspace.test_train_dataset, allow_pickle=True)
            self.train_data = np.pad(self.train_data, [(0, size - side % size) for size, side in zip(self.size, self.train_data.shape)])
            self.target_data = np.load(config.deepspace.test_target_dataset, allow_pickle=True)
            self.target_data = np.pad(self.target_data, [(0, size - side % size) for size, side in zip(self.size, self.target_data.shape)])
        else:
            self.train_data = shared_array[0]
            self.target_data = shared_array[1]
        self.data_shape = self.train_data.shape
        coordinates = []
        for index in range(3):
            coordinate = np.linspace(0, self.train_data.shape[index], int(self.train_data.shape[index]/config.deepspace.image_size[index]) + 1)
            coordinates.append(coordinate[0:-1])
        coordinates = np.meshgrid(coordinates[0], coordinates[1], coordinates[2])
        self.coordinates = np.stack(map(np.ravel, coordinates), axis=1)
        self.coordinates = self.coordinates.astype(np.int)
        self.patcher = IndexPatch(size=config.deepspace.image_size)

    def __getitem__(self, index):
        """return png images

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        coordinate = self.coordinates[index]
        with torch.no_grad():
            train_data = np.copy(self.patcher(coordinate, self.train_data))
            target_data = np.copy(self.patcher(coordinate, self.target_data))
        if self.transform is not None:
            train_data = self.transform(train_data)
            target_data = self.transform(target_data)
        return train_data, target_data, coordinate

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return len(self.coordinates)


class Loader:
    def __init__(self, shared_array=None):
        # process data first
        if isinstance(shared_array, list):
            self.train_data = shared_array[0]
            self.target_data = shared_array[1]
        # transform
        self.train_trainsform = standard_transforms.Compose([
            ToTensor(add_dim=False),
            # RandomRotate(),
        ])
        self.target_transform = standard_transforms.Compose([
            ToTensor(),
        ])
        self.test_transform = standard_transforms.Compose([
            ToTensor(),
        ])
        if config.deepspace.mode == 'train':
            # dataset
            train_set = NPYImages(train_data=self.train_data, target_data=self.target_data, data_length=config.deepspace.train_data_length, train_transform=self.train_trainsform, target_transform=self.target_transform)
            valid_set = NPYTestImages(transform=self.test_transform, shared_array=shared_array)
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
                shuffle=False,
                num_workers=config.deepspace.data_loader_workers
            )
            self.train_iterations = (len(train_set) + config.deepspace.train_batch) // config.deepspace.train_batch
            self.valid_iterations = (len(valid_set) + config.deepspace.validate_batch) // config.deepspace.validate_batch
            self.data_shape = valid_set.data_shape

        elif config.deepspace.mode == 'test':
            test_set = NPYTestImages(transform=self.test_transform, shared_array=None)
            self.test_loader = DataLoader(test_set, batch_size=config.deepspace.test_batch, shuffle=False, num_workers=config.deepspace.data_loader_workers)
            self.test_iterations = (len(test_set) + config.deepspace.test_batch) // config.deepspace.test_batch
            self.data_shape = test_set.data_shape

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
