"""
Defect images Data loader
"""
import imageio
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as standard_transforms

from deepspace.utils.data import AddGaussianNoise
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
    # metrics directory
    config.swap.metrics = {}
    config.swap.metrics.root = root / 'metrics'
    config.swap.metrics.images = root / 'metrics' / 'images'
    config.swap.metrics.ground_truth = root / 'metrics' / 'ground_truth'

    if config.settings.mode == 'metrics':
        image_paths = list(config.swap[mode].images.glob('*'))
        return image_paths

    nomral_images_path = list(config.swap[mode].normal.glob('**/*.' + config.settings.data_format))
    defect_images_path = list(config.swap[mode].defect.glob('**/*.' + config.settings.data_format))
    ground_truth_images_path = list(config.swap[mode].ground_truth.glob('**/*.' + config.settings.data_format))
    print(len(nomral_images_path), len(defect_images_path), len(ground_truth_images_path))

    return nomral_images_path, defect_images_path, ground_truth_images_path


class DQNImages:
    """metrics images for calculating some metrics of the model.
    """

    def __init__(self, mode, transform=None):
        # get all the image paths
        self.image_paths = get_paths(mode)
        if len(self.image_paths) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        """return defect, normal, and grouth_truth images on the same projection angle

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        # get image path
        image_path = self.image_paths[index]
        # the first character in the filename is the label. 0: normal. else: defect, here we get the filename label and transfer it into 0 for normal and 1 for defect
        label = int(int(image_path.name.split('_')[0]) != 0)
        if label == 0:
            # read in images
            # image = imageio.imread(image_path)
            image = np.zeros((512, 512, 36))
            if self.transform is not None:
                image = self.transform(image)
        else:
            paths = image_path.glob('*')
            images = []
            for path in paths:
                image = imageio.imread(path)
                images.append(image)
            image = np.dstack(images)
            if self.transform is not None:
                image = self.transform(image)

        return image, label, [str(image_path)]

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return len(self.image_paths)


class DefectImages:
    def __init__(self, mode, transform=None):
        # get all the image paths
        self.normal_images_path, self.defect_images_path, self.ground_truth_images_path = get_paths(mode)
        if len(self.normal_images_path) == 0 or len(self.defect_images_path) == 0 or len(self.ground_truth_images_path) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        """return defect, normal, and grouth_truth images on the same projection angle

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        # get image path
        defect_image_path = self.defect_images_path[index]
        normal_image_path = config.swap[self.mode].normal / defect_image_path.name
        # read in images
        defect_image = imageio.imread(defect_image_path)
        normal_image = imageio.imread(normal_image_path)
        if self.transform is not None:
            defect_image = self.transform(defect_image)
            normal_image = self.transform(normal_image)
        if config.settings.mode == 'test':
            # for test mode we will get ground truth to calculate metrics
            ground_truth_image_path = config.swap[self.mode].ground_truth / defect_image_path.parent.name / defect_image_path.name
            ground_truth_image = imageio.imread(ground_truth_image_path)
            if self.transform is not None:
                ground_truth_image = self.transform(ground_truth_image)
            return defect_image, normal_image, ground_truth_image, [str(defect_image_path), str(normal_image_path), str(ground_truth_image_path)]
        else:
            return defect_image, normal_image

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return len(self.defect_images_path)


class DefectDataLoader:
    def __init__(self):
        assert config.settings.mode in ['train', 'test', 'metrics']
        # transforms
        this_transform = [
            standard_transforms.ToTensor(),
        ]
        if config.settings.gaussian_noise:
            this_transform.append(AddGaussianNoise(config.settings.gaussian_mean, config.settings.gaussian_std))
        self.input_transform = standard_transforms.Compose(this_transform)

        if config.settings.mode == 'train':
            # training needs train dataset and validate dataset
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

        elif config.settings.mode == 'metrics':
            metrics_set = DQNImages('metrics', transform=self.input_transform)

            self.metrics_loader = DataLoader(metrics_set, batch_size=config.settings.batch_size, shuffle=False,
                                             num_workers=config.settings.data_loader_workers)
            self.metrics_iterations = (len(metrics_set) + config.settings.batch_size) // config.settings.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
