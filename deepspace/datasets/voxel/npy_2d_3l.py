'''
 ┌───┐   ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐ ┌───┬───┬───┐
 │Esc│   │ F1│ F2│ F3│ F4│ │ F5│ F6│ F7│ F8│ │ F9│F10│F11│F12│ │P/S│S L│P/B│  ┌┐    ┌┐    ┌┐
 └───┘   └───┴───┴───┴───┘ └───┴───┴───┴───┘ └───┴───┴───┴───┘ └───┴───┴───┘  └┘    └┘    └┘
 ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───────┐ ┌───┬───┬───┐ ┌───┬───┬───┬───┐
 │~ `│! 1│@ 2│# 3│$ 4│% 5│^ 6│& 7│* 8│( 9│) 0│_ -│+ =│ BacSp │ │Ins│Hom│PUp│ │N L│ / │ * │ - │
 ├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─────┤ ├───┼───┼───┤ ├───┼───┼───┼───┤
 │ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{ [│} ]│ | \ │ │Del│End│PDn│ │ 7 │ 8 │ 9 │   │
 ├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤ └───┴───┴───┘ ├───┼───┼───┤ + │
 │ Caps │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  │               │ 4 │ 5 │ 6 │   │
 ├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────────┤     ┌───┐     ├───┼───┼───┼───┤
 │ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│  Shift   │     │ ↑ │     │ 1 │ 2 │ 3 │   │
 ├─────┬──┴─┬─┴──┬┴───┴───┴───┴───┴───┴──┬┴───┼───┴┬────┬────┤ ┌───┼───┼───┐ ├───┴───┼───┤ E││
 │ Ctrl│    │Alt │         Space         │ Alt│    │    │Ctrl│ │ ← │ ↓ │ → │ │   0   │ . │←─┘│
 └─────┴────┴────┴───────────────────────┴────┴────┴────┴────┘ └───┴───┴───┘ └───────┴───┴───┘

Description: Defect images Data loader, for training on normal images
Author: Zhihua Liang
Github: https://github.com/magicknight
Date: 2021-08-12 19:38:41
LastEditors: Zhihua Liang
LastEditTime: 2021-08-21 17:32:18
FilePath: /home/zhihua/framework/deepspace/deepspace/datasets/voxel/npy_2d_3l.py
'''


import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms.transforms import ColorJitter, RandomAdjustSharpness, RandomAutocontrast
from deepspace.augmentation.image import CornerErasing, RandomErasing, RandomBreak, AddGaussianNoise, RandomRotation
from commontools.setup import config


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        data = torch.from_numpy(data)
        if data.dim() == 2:
            data = data.unsqueeze(0)
        return data


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
        train_data = np.copy(self.train_data[index:index+3, :, :])
        target_data = np.copy(self.target_data[index+1, :, :])
        if self.train_transform is not None:
            train_data = self.train_transform(train_data)
        if self.target_transform is not None:
            target_data = self.target_transform(target_data)
        # orientation = np.random.randint(0, config.deepspace.rotation_node)
        # train_data, _ = RandomRotation(train_data, orientation)
        # target_data, _ = RandomRotation(target_data, orientation)

        return train_data, target_data

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return self.train_data.shape[0] - 2


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
        real_index = np.random.randint(0, self.train_data.shape[0]-2)
        train_data = np.copy(self.train_data[real_index:real_index+3, :, :])
        target_data = np.copy(self.target_data[real_index+1, :, :])
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
            self.input_data = np.load(config.deepspace.test_dataset, allow_pickle=True)
            # self.input_data = np.pad(self.input_data, [(0, 0), (0, config.deepspace.image_size[0] - self.input_data[1]), (0, config.deepspace.image_size[1] - self.input_data[2])])
        self.input_transform = transform

    def __getitem__(self, index):
        """return png images

        Args:
            index (int): data index

        Returns:
            Tensor: images
        """
        input_data = self.input_data[index:index+3, :, :]
        if self.input_transform is not None:
            input_data = self.input_transform(input_data)

        return input_data, index

    def __len__(self):
        # the size defect images is the size of this dataset, not the size of normal images
        return self.input_data.shape[0] - 2


class Loader:
    def __init__(self, shared_array=None):
        if isinstance(shared_array, list):
            self.train_data = shared_array[0]
            self.target_data = shared_array[1]
        if shared_array is None:
            if config.deepspace.train_dataset.startswith('gs://'):
                from torch_xla.utils.gcsfs import read
                self.train_data = np.frombuffer(read(config.deepspace.train_dataset), dtype=config.deepspace.dataset_type)[32:].reshape(config.deepspace.dataset_shape)
                self.target_data = np.frombuffer(read(config.deepspace.target_dataset), dtype=config.deepspace.dataset_type)[32:].reshape(config.deepspace.dataset_shape)
            else:
                self.train_data = np.load(config.deepspace.train_dataset, allow_pickle=True)
                self.target_data = np.load(config.deepspace.target_dataset, allow_pickle=True)
        # transform
        self.train_trainsform = transforms.Compose([
            ToTensor(),
            # transforms.CenterCrop(config.deepspace.image_size),
            # transforms.Resize(config.deepspace.image_resolution),
            # CornerErasing(config.deepspace.image_resolution/2)
            # transforms.RandomInvert(0.5),
            # transforms.RandomAutocontrast(0.5),
            # transforms.RandomAdjustSharpness(0.5),
            # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            # RandomBreak(
            #     probability=config.deepspace.break_probability,
            #     sl=config.deepspace.min_erasing_area,
            #     sh=config.deepspace.max_erasing_area,
            #     value=config.deepspace.value,
            #     return_normal=False,
            # )
            # AddGaussianNoise(mean=config.deepspace.noise_mean, std=config.deepspace.noise_std),
        ])
        self.target_transform = transforms.Compose([
            ToTensor(),
        ])
        self.test_transform = transforms.Compose([
            ToTensor(),
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
                    shuffle=config.deepspace.shuffle,
                )
                validate_sampler = DistributedSampler(
                    valid_set,
                    num_replicas=xla_model.xrt_world_size(),
                    rank=xla_model.get_ordinal(),
                    shuffle=config.deepspace.shuffle,
                )
            else:
                train_sampler = None
                validate_sampler = None
            # training needs train dataset and validate dataset
            self.train_loader = DataLoader(
                train_set,
                batch_size=config.deepspace.train_batch,
                sampler=train_sampler,
                drop_last=config.deepspace.drop_last,
                num_workers=config.deepspace.data_loader_workers
            )
            self.valid_loader = DataLoader(
                valid_set,
                batch_size=config.deepspace.validate_batch,
                sampler=validate_sampler,
                drop_last=config.deepspace.drop_last,
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
