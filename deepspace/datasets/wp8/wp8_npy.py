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


def get_paths():
    """return paths of images

    Args:

    Raises:
        Exception: proper mode

    Returns:
        list: items list, and masks list if in test mode
    """
    root = Path(config.deepspace.dataset_root)
    # return images path
    images_path = list(config.swap.root.glob('**/*.' + config.deepspace.data_format))
    return images_path


class NPYImages:
    def __init__(self, transform=None):
        # get all the image paths
        self.images_path = get_paths()
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
        # read in images
        image = np.load(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return len(self.images_path)


class NPYDataLoader:
    def __init__(self):
        # transform
        self.input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
        ])
        # split dataset
        data_set = NPYImages(transform=self.input_transform)
        train_size = int(0.9 * len(data_set))
        valid_size = len(data_set) - train_size
        train_set, valid_set = torch.utils.data.random_split(data_set, [train_size, valid_size], generator=torch.Generator().manual_seed(config.deepspace.seed))
        if config.deepspace.mode == 'train':
            # training needs train dataset and validate dataset
            self.train_loader = DataLoader(train_set, batch_size=config.deepspace.batch_size, shuffle=True,
                                           num_workers=config.deepspace.data_loader_workers)
            self.valid_loader = DataLoader(valid_set, batch_size=config.deepspace.batch_size, shuffle=False,
                                           num_workers=config.deepspace.data_loader_workers)
            self.train_iterations = (len(train_set) + config.deepspace.batch_size) // config.deepspace.batch_size
            self.valid_iterations = (len(valid_set) + config.deepspace.batch_size) // config.deepspace.batch_size

        elif config.deepspace.mode == 'test':
            train_size = int(0.8 * len(data_set))
            test_size = int(0.2 * len(data_set))
            train_set, test_set = torch.utils.data.random_split(data_set, [train_size, test_size])
            self.test_loader = DataLoader(test_set, batch_size=config.deepspace.batch_size, shuffle=False,
                                          num_workers=config.deepspace.data_loader_workers)
            self.test_iterations = (len(test_set) + config.deepspace.batch_size) // config.deepspace.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
