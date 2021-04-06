"""
Defect images Data loader, for training on normal images
"""
import imageio
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms

from deepspace.augmentation.image import CornerErasing, RandomErasing, RandomBreak

from commontools.setup import config, logger


def get_paths(mode):
    """return paths of images

    Args:
        mode (string): dataloader mode, like 'train', 'test', 'validate'

    Raises:
        Exception: proper mode

    Returns:
        list: items list, and masks list if in test mode
    """
    root = Path(config.deepspace.dataset_root)
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
    images_path = list(config.swap[mode].root.glob('**/*.' + config.deepspace.data_format))
    return images_path


class RingImages:
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
            broken_image, image = self.transform(image)
            return broken_image, image
        else:
            return image, image

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return len(self.images_path)


class BreakDataLoader:
    def __init__(self, center_crop=None):
        assert config.deepspace.mode in ['train', 'test', 'real']

        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(config.deepspace.image_resolution),
            # CornerErasing(config.deepspace.image_resolution/2)
            RandomBreak(
                probability=config.deepspace.probability,
                sl=config.deepspace.min_erasing_area,
                sh=config.deepspace.max_erasing_area,
                value=config.deepspace.value,
            )
        ])

        if config.deepspace.mode == 'train':
            # training needs train dataset and validate dataset
            train_set = RingImages('train', transform=self.input_transform)
            valid_set = RingImages('validate', transform=self.input_transform)
            self.train_loader = DataLoader(train_set, batch_size=config.deepspace.batch_size, shuffle=True,
                                           num_workers=config.deepspace.data_loader_workers)
            self.valid_loader = DataLoader(valid_set, batch_size=config.deepspace.batch_size, shuffle=False,
                                           num_workers=config.deepspace.data_loader_workers)
            self.train_iterations = (len(train_set) + config.deepspace.batch_size) // config.deepspace.batch_size
            self.valid_iterations = (len(valid_set) + config.deepspace.batch_size) // config.deepspace.batch_size

        elif config.deepspace.mode == 'test':
            test_set = RingImages('test', transform=self.input_transform)

            self.test_loader = DataLoader(test_set, batch_size=config.deepspace.batch_size, shuffle=False,
                                          num_workers=config.deepspace.data_loader_workers)
            self.test_iterations = (len(test_set) + config.deepspace.batch_size) // config.deepspace.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
