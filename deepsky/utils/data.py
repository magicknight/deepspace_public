import numpy as np
from imageio import imwrite


def save_images(data, paths):
    """a helper function to save images

    Args:
        data (ndarray): a numpy array with n x w x h shape
        paths (list): a list of paths
    """
    for image, path in zip(data, paths):
        imwrite(path, image)


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
    ct_data = ct_data.transpose([1, 2, 0])
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
