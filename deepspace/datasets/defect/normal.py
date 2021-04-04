"""
Defect images Data loader, for training on normal images
"""
import imageio
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as standard_transforms

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
    config.swap.train.normal = root / 'train' / 'normal'
    config.swap.train.defect = root / 'train' / 'defect'
    config.swap.train.ground_truth = root / 'train' / 'ground_truth'
    # validate directory
    config.swap.validate = {}
    config.swap.validate.root = root / 'validate'
    config.swap.validate.normal = root / 'validate' / 'normal'
    config.swap.validate.defect = root / 'validate' / 'defect'
    config.swap.validate.ground_truth = root / 'validate' / 'ground_truth'
    # test directory
    config.swap.test = {}
    config.swap.test.root = root / 'test'
    config.swap.test.normal = root / 'test' / 'normal'
    config.swap.test.defect = root / 'test' / 'defect'
    config.swap.test.ground_truth = root / 'test' / 'ground_truth'

    nomral_images_path = list(config.swap[mode].normal.glob('**/*.' + config.deepspace.data_format))
    defect_images_path = list(config.swap[mode].defect.glob('**/*.' + config.deepspace.data_format))
    ground_truth_images_path = list(config.swap[mode].ground_truth.glob('**/*.' + config.deepspace.data_format))

    return nomral_images_path, defect_images_path, ground_truth_images_path


class DefectImages:
    def __init__(self, mode, transform=None):
        # get all the image paths
        self.normal_images_path, self.defect_images_path, self.ground_truth_images_path = get_paths(mode)
        # if len(self.normal_images_path) == 0 or len(self.defect_images_path) == 0 or len(self.ground_truth_images_path) == 0:
        #     raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        """return defect, normal, and grouth_truth images on the same projection angle

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        if config.deepspace.mode == 'test':
            defect_image_path = self.defect_images_path[index]
            defect_image = imageio.imread(defect_image_path)
            # for test mode we will get ground truth to calculate metrics
            ground_truth_image_path = config.swap[self.mode].ground_truth / defect_image_path.name
            ground_truth_image = imageio.imread(ground_truth_image_path)
            if self.transform is not None:
                ground_truth_image = self.transform(ground_truth_image)
                defect_image = self.transform(defect_image)
            return defect_image, ground_truth_image
        else:
            # get image path
            normal_image_path = self.normal_images_path[index]
            # read in images
            normal_image = imageio.imread(normal_image_path)
            if self.transform is not None:
                normal_image = self.transform(normal_image)
            return normal_image

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        if config.deepspace.mode == 'test':
            return len(self.defect_images_path)
        else:
            return len(self.normal_images_path)


class DefectDataLoader:
    def __init__(self):
        assert config.deepspace.mode in ['train', 'test']

        self.input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
        ])

        if config.deepspace.mode == 'train':
            # training needs train dataset and validate dataset
            train_set = DefectImages('train', transform=self.input_transform)
            valid_set = DefectImages('validate', transform=self.input_transform)
            self.train_loader = DataLoader(train_set, batch_size=config.deepspace.batch_size, shuffle=True,
                                           num_workers=config.deepspace.data_loader_workers)
            self.valid_loader = DataLoader(valid_set, batch_size=config.deepspace.batch_size, shuffle=False,
                                           num_workers=config.deepspace.data_loader_workers)
            self.train_iterations = (len(train_set) + config.deepspace.batch_size) // config.deepspace.batch_size
            self.valid_iterations = (len(valid_set) + config.deepspace.batch_size) // config.deepspace.batch_size

        elif config.deepspace.mode == 'real' or config.deepspace.mode == 'test':
            test_set = DefectImages('test', transform=self.input_transform)

            self.test_loader = DataLoader(test_set, batch_size=config.deepspace.batch_size, shuffle=False,
                                          num_workers=config.deepspace.data_loader_workers)
            self.test_iterations = (len(test_set) + config.deepspace.batch_size) // config.deepspace.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
