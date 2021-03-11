"""
Defect images Data loader
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
    assert mode in ['train', 'val', 'validate', 'test', 'inference']
    root = Path(config.settings.dataset_root)
    items = []
    if mode == 'train':
        image_path = root / 'train' / 'good'
    elif mode == 'val' or mode == 'validate':
        image_path = root / 'validate' / 'good'
    elif mode == 'test' or mode == 'inference':
        image_path = root / 'test'
        ground_truth_path = root / 'ground_truth'
    else:
        raise Exception("Please choose proper mode for data")
    items = list(image_path.glob('**/*.' + config.settings.data_format))
    if mode == 'test':
        ground_truth_items = list(ground_truth_path.glob('**/*.' + config.settings.data_format))
    else:
        ground_truth_items = None

    return items, ground_truth_items


class DefectImages:
    def __init__(self, mode, transform=None):
        self.imgs, self.masks = get_paths(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.imgs[index]
            image = imageio.imread(img_path)
            if self.transform is not None:
                image = self.transform(image)
            return image

        if self.mode == 'validate':
            img_path = self.imgs[index]
            image = imageio.imread(img_path)
            if self.transform is not None:
                image = self.transform(image)
            return image

        if self.mode == 'test':
            img_path = self.imgs[index]
            image = imageio.imread(img_path)
            mask_path = self.masks[index]
            mask = imageio.imread(mask_path)
            if self.transform is not None:
                image = self.transform(image)
                mask = self.transform(mask)
            return image, mask

    def __len__(self):
        return len(self.imgs)


class DefectDataLoader:
    def __init__(self):
        assert config.settings.mode in ['train', 'test', 'validate']

        self.input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
        ])

        if config.settings.mode == 'train':
            train_set = DefectImages('train', transform=self.input_transform)
            valid_set = DefectImages('validate', transform=self.input_transform)

            self.train_loader = DataLoader(train_set, batch_size=config.settings.batch_size, shuffle=True,
                                           num_workers=config.settings.data_loader_workers)
            self.valid_loader = DataLoader(valid_set, batch_size=config.settings.batch_size, shuffle=False,
                                           num_workers=config.settings.data_loader_workers)
            self.train_iterations = (len(train_set) + config.settings.batch_size) // config.settings.batch_size
            self.valid_iterations = (len(valid_set) + config.settings.batch_size) // config.settings.batch_size

        elif config.settings.mode == 'test':
            test_set = DefectImages('test', transform=self.input_transform)

            self.test_loader = DataLoader(test_set, batch_size=config.settings.batch_size, shuffle=False,
                                          num_workers=config.settings.data_loader_workers)
            self.test_iterations = (len(test_set) + config.settings.batch_size) // config.settings.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
