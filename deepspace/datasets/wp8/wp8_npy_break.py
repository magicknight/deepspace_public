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


def get_paths(path):
    """return paths of images

    Args:

    Raises:
        Exception: proper mode

    Returns:
        list: items list, and masks list if in test mode
    """
    root = Path(path)
    # return images path
    data_path = list(root.glob('**/*.' + config.deepspace.data_format))
    return data_path


class NPYImages:
    def __init__(self, path, train_transform=None, target_transform=None):
        # get all the image paths
        self.data_path = get_paths(path)
        self.train_transform = train_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """return data

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        # get image path
        data_path = self.data_path[index]

        # read in images data.shape (64, 64, 64)
        train_data = np.load(data_path)
        # target data is non-break train data
        if self.target_transform is not None:
            target_data = self.target_transform(train_data)
        if self.train_transform is not None:
            train_data = self.train_transform(train_data)
        return train_data, target_data

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return len(self.data_path)


class NPYImagesTest:
    def __init__(self, path, transform=None):
        # get all the image paths
        self.data_path = get_paths(path)
        self.transform = transform

    def __getitem__(self, index):
        """return data

        Args:
            index (int): data index

        Returns:
            Tensor: data
        """
        # get image path
        data_path = self.data_path[index]

        # read in data data.shape (30, 30, 30)
        data = np.load(data_path)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        # the size defect data is the size of this dataset, not the size of normal data
        return len(self.data_path)


class NPYDataLoader:
    def __init__(self):
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
        # split dataset
        train_dataset = NPYImages(path=config.deepspace.train_dataset, train_transform=self.train_trainsform, target_transform=self.target_transform)
        train_size = int(config.deepspace.split_rate * len(train_dataset))
        valid_size = len(train_dataset) - train_size
        train_set, valid_set = torch.utils.data.random_split(train_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(config.deepspace.seed))
        # training needs train dataset and validate dataset
        self.train_loader = DataLoader(train_set, batch_size=config.deepspace.train_batch, shuffle=True,
                                       num_workers=config.deepspace.data_loader_workers)
        self.valid_loader = DataLoader(valid_set, batch_size=config.deepspace.validate_batch, shuffle=False,
                                       num_workers=config.deepspace.data_loader_workers)
        self.train_iterations = (len(train_set) + config.deepspace.train_batch) // config.deepspace.train_batch
        self.valid_iterations = (len(valid_set) + config.deepspace.validate_batch) // config.deepspace.validate_batch

        test_dataset = NPYImagesTest(path=config.deepspace.test_dataset, transform=self.test_transform)
        self.test_loader = DataLoader(test_dataset, batch_size=config.deepspace.test_batch, shuffle=False,
                                      num_workers=config.deepspace.data_loader_workers)
        self.test_iterations = (len(test_dataset) + config.deepspace.test_batch) // config.deepspace.test_batch

    def finalize(self):
        pass
