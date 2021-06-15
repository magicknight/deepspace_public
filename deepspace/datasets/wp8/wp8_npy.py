"""
Defect images Data loader, for training on normal images
"""
from tokenize import generate_tokens
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as standard_transforms

from commontools.setup import config, logger


class ToTensor(object):
    """Convert 3D ndarrays in sample to Tensors."""

    def __call__(self, image):
        # # swap color axis because
        # # numpy image: H x W x C
        # # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        return image


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
    images_path = list(root.glob('**/*.' + config.deepspace.data_format))
    return images_path


class NPYImages:
    def __init__(self, path, transform=None):
        # get all the image paths
        self.images_path = get_paths(path)
        self.transform = transform

    def __getitem__(self, index):
        """return png images

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        # get image path
        image_path = self.images_path[index]

        # read in images data.shape (30, 30, 30)
        image = np.load(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, str(image_path)

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return len(self.images_path)


class NPYDataLoader:
    def __init__(self):
        # transform
        self.input_transform = standard_transforms.Compose([
            ToTensor(),
        ])
        # split dataset
        train_dataset = NPYImages(path=config.deepspace.train_dataset, transform=self.input_transform)
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

        test_dataset = NPYImages(path=config.deepspace.test_dataset, transform=self.input_transform)
        self.test_loader = DataLoader(test_dataset, batch_size=config.deepspace.test_batch, shuffle=False,
                                      num_workers=config.deepspace.data_loader_workers)
        self.test_iterations = (len(test_dataset) + config.deepspace.test_batch) // config.deepspace.test_batch

    def finalize(self):
        pass
