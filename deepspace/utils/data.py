import numpy as np
import torch
from imageio import imwrite
from numpy.lib.function_base import diff
from seaborn import heatmap
from matplotlib import pyplot as plt
from tqdm import tqdm


def save_images(data, paths, is_tqdm=False):
    """a helper function to save images

    Args:
        data (ndarray): a numpy array with n x w x h shape
        paths (list): a list of paths
    """
    if is_tqdm:
        images_and_paths = tqdm(zip(data, paths), desc='data sets', total=len(paths))
    else:
        images_and_paths = zip(data, paths)
    for image, path in images_and_paths:
        imwrite(path, image)


def make_heatmaps(output_images, images, paths):
    """plot heatmaps of two images

    Args:
        output_images (np array): image 1
        images (np array ): image 2
        paths (string or pathlib Path): a path to save the heatmap
    """
    # diff_images = normalization(output - images)
    diff_images = output_images - images
    for diff_image, path in zip(diff_images, paths):
        diff_heatmap = heatmap(diff_image).get_figure()
        plt.axis('off')
        diff_heatmap.savefig(path)
        plt.close()


def make_masks(images, threshold=None):
    """make masks for images and comparation images

    Args:
        images (ndarray): images
        compare_images (ndarray): comparation images
        threshold (int, optional): a threshold to cut. Defaults to None.

    Returns:
        ndarray: masks images
    """
    zeros = np.zeros_like(images[0], dtype=np.uint8)
    ones = np.ones_like(images[0], dtype=np.uint8)
    if threshold is None:
        threshold = np.median(images)
    masks = np.where(images < threshold, ones, zeros)
    # mask = mask.astype(np.uint8)
    return masks


def make_masks_tensor(images, device, threshold=None):
    """make masks for images and comparation images

    Args:
        images (ndarray): images
        compare_images (ndarray): comparation images
        threshold (int, optional): a threshold to cut. Defaults to None.

    Returns:
        ndarray: masks images
    """
    zeros = torch.zeros_like(images[0], device=device, dtype=torch.float)
    ones = torch.ones_like(images[0], device=device, dtype=torch.float)
    if threshold is None:
        threshold = torch.median(images)
    masks = torch.where(images < threshold, ones, zeros)
    # mask = mask.astype(np.uint8)
    return masks


def read_numpy(path, shape, datatype=np.float32) -> np.array:
    """read in numpy binary data

    Args:
        path (string or pathlib Path): file path
        shape (tuple): array shape
        datatype ([type], optional): [description]. Defaults to np.float32.

    Returns:
        np.array: [description]
    """

    with open(path, mode='rb') as file:
        ct_data = np.fromfile(file, datatype)
    ct_data = ct_data.reshape(shape)
    # ct_data = ct_data.transpose([1, 2, 0])
    return ct_data


def normalization(image, epsilon=1e-9):
    """
    projection to logs

    Args:
        projection ([type]): [description]
        epsilon ([type], optional): [description]. Defaults to 1e-9.

    Returns:
        [type]: [description]
    """
    this_range = np.max(image) - np.min(image)
    if this_range < epsilon:
        this_range = epsilon
    image = (image - np.min(image)) / this_range
    image = 1 - image
    # image = image * 255
    return image


def to_uint8(images) -> np.array:
    """prepare to write to png file. turns float to uint8

    Args:
        images (np.array): nd array

    Returns:
        np.array: np array in uint8 type
    """
    images = normalization(images) * 255
    images = images.astype(np.uint8)
    return images
