"""
The Oxford-IIIT Pet Dataset

We have created a 37 category pet dataset with roughly 200 images for each class. 
The images have a large variations in scale, pose and lighting. 
All images have an associated ground truth annotation of breed, head ROI, and pixel level trimap segmentation.

https://www.robots.ox.ac.uk/~vgg/data/pets/
"""
import re
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor
import PIL

from commontools.setup import config, logger


def extract_label(fname):
    # Function to get the label from the filename
    stem = fname.split(os.path.sep)[-1]
    return re.search(r"^(.*)_\d+\.jpg$", stem).groups()[0]


class PetsDataset(Dataset):
    def __init__(self, file_names, image_transform=None, label_to_id=None):
        self.file_names = file_names
        self.image_transform = image_transform
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        raw_image = PIL.Image.open(fname)
        image = raw_image.convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        label = extract_label(fname)
        if self.label_to_id is not None:
            label = self.label_to_id[label]
        return {"image": image, "label": label}


def get_data_loader():
    # Grab all the image filenames
    file_names = [os.path.join(config.deepspace.dataset_root, fname) for fname in os.listdir(config.deepspace.dataset_root) if fname.endswith(config.deepspace.data_format)]
    # Build the label correspondences
    all_labels = [extract_label(fname) for fname in file_names]
    id_to_label = list(set(all_labels))
    id_to_label.sort()
    label_to_id = {lbl: i for i, lbl in enumerate(id_to_label)}

    num_classes = len(label_to_id)

    # Set the seed before splitting the data.
    np.random.seed(config.deepspace.seed)
    torch.manual_seed(config.deepspace.seed)
    torch.cuda.manual_seed_all(config.deepspace.seed)

    # Split our filenames between train and validation
    random_perm = np.random.permutation(len(file_names))
    cut = int(0.8 * len(file_names))
    train_split = random_perm[:cut]
    eval_split = random_perm[cut:]

    image_size = config.deepspace.image_size
    if not isinstance(image_size, (list, tuple)):
        image_size = (image_size, image_size)
        # For training we use a simple RandomResizedCrop
    train_transform = Compose([RandomResizedCrop(image_size, scale=(0.5, 1.0)), ToTensor()])
    train_dataset = PetsDataset([file_names[i] for i in train_split], image_transform=train_transform, label_to_id=label_to_id)
    # For evaluation, we use a deterministic Resize
    validate_transform = Compose([Resize(image_size), ToTensor()])
    eval_dataset = PetsDataset([file_names[i] for i in eval_split], image_transform=validate_transform, label_to_id=label_to_id)

    # Instantiate dataloaders.
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.deepspace.batch_size, num_workers=config.deepspace.data_loader_workers)
    valid_loader = DataLoader(eval_dataset, shuffle=False, batch_size=config.deepspace.batch_size, num_workers=config.deepspace.data_loader_workers)

    return train_loader, valid_loader, num_classes
