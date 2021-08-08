"""
Defect images Data loader, for training on normal images
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms.transforms import ColorJitter, RandomAdjustSharpness, RandomAutocontrast
from deepspace.augmentation.image import CornerErasing, RandomErasing, RandomBreak, RandomRotation
from commontools.setup import config


class TrainDataset(object):
    """
    Load the defect images, and return the data loader
    """

    def __init__(self, train_data, target_data, train_transform=None, target_transform=None):
        # get all the image paths
        self.train_data = train_data
        self.target_data = target_data
        self.train_transform = train_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """return png images

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        train_data = np.copy(self.train_data[index, :, :])
        target_data = np.copy(self.target_data[index, :, :])
        if self.train_transform is not None:
            train_data = self.train_transform(train_data)
        if self.target_transform is not None:
            target_data = self.target_transform(target_data)

        orientation_1 = np.random.randint(0, config.deepspace.rotation_node)
        orientation_2 = np.random.randint(0, config.deepspace.rotation_node)
        train_data_1, _ = RandomRotation(train_data, orientation_1)
        target_data_1, _ = RandomRotation(target_data, orientation_1)
        train_data_2, _ = RandomRotation(train_data, orientation_2)
        target_data_2, _ = RandomRotation(target_data, orientation_2)

        return train_data_1, target_data_1, orientation_1, train_data_2, target_data_2, orientation_2

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return self.train_data.shape[0]


class ValiDataset(object):
    """
    Load the defect images, and return the data loader
    """

    def __init__(self, train_data, target_data, train_transform=None, target_transform=None):
        # get all the image paths
        self.train_data = train_data
        self.target_data = target_data
        self.train_transform = train_transform
        self.target_transform = target_transform
        self.data_len = config.deepspace.validation_size

    def __getitem__(self, index):
        """return png images

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        real_index = np.random.randint(0, self.train_data.shape[0])
        train_data = np.copy(self.train_data[real_index, :, :])
        target_data = np.copy(self.target_data[real_index, :, :])
        if self.train_transform is not None:
            train_data = self.train_transform(train_data)
        if self.target_transform is not None:
            target_data = self.target_transform(target_data)
        return train_data, target_data

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return self.data_len


class TestDataset(object):
    """
    Load the defect images, and return the data loader
    """

    def __init__(self, transform=None, shared_array=None):
        # get all the image paths
        self.size = config.deepspace.image_size
        if shared_array is None:
            self.input_data = np.load(config.deepspace.test_train_dataset, allow_pickle=True)
            self.input_data = np.pad(self.input_data, [(0, 0), (0, config.deepspace.image_size[0] - self.input_data[1]), (0, config.deepspace.image_size[1] - self.input_data[2])])
        self.input_transform = transform

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
        # transform
        self.train_trainsform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop(config.deepspace.image_size),
            # transforms.Resize(config.deepspace.image_resolution),
            # CornerErasing(config.deepspace.image_resolution/2)
            transforms.RandomInvert(0.5),
            transforms.RandomAutocontrast(0.5),
            transforms.RandomAdjustSharpness(0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            RandomBreak(
                probability=config.deepspace.break_probability,
                sl=config.deepspace.min_erasing_area,
                sh=config.deepspace.max_erasing_area,
                value=config.deepspace.value,
                return_normal=False,
            )
        ])
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if config.deepspace.mode == 'train':
            train_set = TrainDataset(train_data=self.train_data, target_data=self.target_data, train_transform=self.train_trainsform, target_transform=self.target_transform)
            valid_set = ValiDataset(train_data=self.train_data, target_data=self.target_data, train_transform=self.train_trainsform, target_transform=self.target_transform)
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
                shuffle=config.deepspace.shuffle,
                num_workers=config.deepspace.data_loader_workers
            )
            self.train_iterations = len(train_set) // config.deepspace.train_batch
            self.valid_iterations = len(valid_set) // config.deepspace.validate_batch

        elif config.deepspace.mode == 'test':
            test_set = TestDataset(transform=self.test_transform, shared_array=None)
            self.test_loader = DataLoader(test_set, batch_size=config.deepspace.test_batch, shuffle=False, num_workers=config.deepspace.data_loader_workers)
            self.test_iterations = len(test_set) // config.deepspace.test_batch
        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
