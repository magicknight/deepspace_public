"""
imagenet dataloader

"""
import imageio
from pathlib import Path
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
import torch_xla.core.xla_model as xm

from commontools.setup import config, logger


class Loader:
    def __init__(self):
        normalize = transforms.Normalize(
            mean=config.deepspace.mean, std=config.deepspace.std)

        train_dataset = torchvision.datasets.ImageFolder(
            Path(config.deepspace.dataset) / 'train',
            transforms.Compose([
                transforms.RandomResizedCrop(config.deepspace.image_size),
                transforms.Resize(config.deepspace.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        validate_dataset = torchvision.datasets.ImageFolder(
            Path(config.deepspace.dataset) / 'val',
            # Matches Torchvision's eval transforms except Torchvision uses size
            # 256 resize for all models both here and in the train loader. Their
            # version crashes during training on 299x299 images, e.g. inception.
            transforms.Compose([
                transforms.Resize(config.deepspace.image_size),
                transforms.CenterCrop(config.deepspace.image_size),
                transforms.ToTensor(),
                normalize,
            ]))

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)
        validate_sampler = torch.utils.data.distributed.DistributedSampler(
            validate_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False)

        if config.deepspace.mode == 'train':
            # training needs train dataset and validate dataset
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=config.deepspace.train_batch,
                sampler=train_sampler,
                drop_last=config.deepspace.drop_last,
                shuffle=False,
                num_workers=config.deepspace.data_loader_workers
            )
            self.valid_loader = DataLoader(
                validate_dataset,
                batch_size=config.deepspace.validate_batch,
                sampler=validate_sampler,
                drop_last=config.deepspace.drop_last,
                shuffle=False,
                num_workers=config.deepspace.data_loader_workers
            )
            self.train_iterations = (len(train_dataset) + config.deepspace.train_batch) // config.deepspace.train_batch
            self.valid_iterations = (len(validate_sampler) + config.deepspace.validate_batch) // config.deepspace.validate_batch

        elif config.deepspace.mode == 'test':
            test_dataset = torchvision.datasets.ImageFolder(
                Path(config.deepspace.dataset) / 'test',
                # Matches Torchvision's eval transforms except Torchvision uses size
                # 256 resize for all models both here and in the train loader. Their
                # version crashes during training on 299x299 images, e.g. inception.
                transforms.Compose([
                    transforms.Resize(config.deepspace.image_size),
                    # transforms.CenterCrop(config.deepspace.image_size),
                    transforms.ToTensor(),
                    normalize,
                ]))
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=False)
            self.test_loader = DataLoader(
                validate_dataset,
                batch_size=config.deepspace.test_batch,
                sampler=validate_sampler,
                shuffle=False,
                num_workers=config.deepspace.data_loader_workers
            )
            self.test_iterations = (len(test_set) + config.deepspace.batch_size) // config.deepspace.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
