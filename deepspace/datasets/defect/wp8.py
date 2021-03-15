"""
Defect images Data loader, for training on normal images
"""
import imageio
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as standard_transforms

from deepspace.config.config import config, logger


def get_paths(mode):
    """return paths of images

    Args:
        mode (string): dataloader mode, like 'train', 'test', 'validate'

    Raises:
        Exception: proper mode

    Returns:
        list: items list, and masks list if in test mode
    """
    root = Path(config.settings.dataset_root)
    # train directory
    config.swap.train = {}
    config.swap.train.root = root / 'train'
    # validate directory
    config.swap.validate = {}
    config.swap.validate.root = root / 'validate'
    # test directory
    config.swap.test = {}
    config.swap.test.root = root / 'test'
    # return images path
    images_path = list(config.swap[mode].root.glob('**/*.' + config.settings.data_format))
    return images_path


class PNGImages:
    def __init__(self, mode, transform=None):
        # get all the image paths
        self.images_path = get_paths(mode)
        self.mode = mode
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
        image = imageio.imread(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return len(self.images_path)


class PNGDataLoader:
    def __init__(self):
        assert config.settings.mode in ['train', 'test', 'real']

        self.input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Resize(config.settings.image_resolution),
        ])

        if config.settings.mode == 'train':
            # training needs train dataset and validate dataset
            train_set = PNGImages('train', transform=self.input_transform)
            valid_set = PNGImages('validate', transform=self.input_transform)
            self.train_loader = DataLoader(train_set, batch_size=config.settings.batch_size, shuffle=True,
                                           num_workers=config.settings.data_loader_workers)
            self.valid_loader = DataLoader(valid_set, batch_size=config.settings.batch_size, shuffle=False,
                                           num_workers=config.settings.data_loader_workers)
            self.train_iterations = (len(train_set) + config.settings.batch_size) // config.settings.batch_size
            self.valid_iterations = (len(valid_set) + config.settings.batch_size) // config.settings.batch_size

        elif config.settings.mode == 'real' or config.settings.mode == 'test':
            test_set = PNGImages('test', transform=self.input_transform)

            self.test_loader = DataLoader(test_set, batch_size=config.settings.batch_size, shuffle=False,
                                          num_workers=config.settings.data_loader_workers)
            self.test_iterations = (len(test_set) + config.settings.batch_size) // config.settings.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
