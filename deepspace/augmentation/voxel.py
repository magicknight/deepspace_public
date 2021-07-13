import random
import math
import numpy as np
from torchvision import transforms
import torch


class RandomBreak(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sides_range: sides range
    value: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sides_range=[[0.2, 0.7], [0.2, 0.7], [0.2, 0.7]], value=0):
        self.probability = probability
        self.value = value
        self.sides_range = np.array(sides_range)

    def __call__(self, data_3d):

        if random.uniform(0, 1) > self.probability:
            return data_3d

        heigh = int(round(random.uniform(self.sides_range[0, 0]*data_3d.shape[1], self.sides_range[0, 1]*data_3d.shape[1])))
        width = int(round(random.uniform(self.sides_range[0, 0]*data_3d.shape[2], self.sides_range[0, 1]*data_3d.shape[2])))
        depth = int(round(random.uniform(self.sides_range[0, 0]*data_3d.shape[3], self.sides_range[0, 1]*data_3d.shape[3])))

        init_x = random.randint(0, data_3d.shape[1] - heigh)
        init_y = random.randint(0, data_3d.shape[2] - width)
        init_z = random.randint(0, data_3d.shape[3] - depth)
        # print('==============================')
        # print(init_x, init_y, init_z, heigh, width, depth)
        # print('before', data_3d.shape, data_3d.min(), data_3d.max(), data_3d.mean())
        data_3d[:, init_x:init_x+heigh, init_y:init_y+width, init_z:init_z+depth] = self.value
        # print('after', data_3d.shape, data_3d.min(), data_3d.max(), data_3d.mean())

        return data_3d


class IndexPatch(object):
    """sample 3D tensor"""

    def __init__(self, size=64):
        if isinstance(size, (list)):
            self.size = size
        else:
            self.size = [size] * 3

    def __call__(self, index, data_3d):
        patch = torch.zeros(self.size)

        init_x = index[0]
        init_y = index[1]
        init_z = index[2]

        patch = data_3d[init_x:init_x + self.size[0], init_y:init_y + self.size[1], init_z:init_z + self.size[2]]

        return patch


class RandomPatch(object):
    """sample 3D tensor"""

    def __init__(self, size=64):
        if isinstance(self.size, (list)):
            self.size = [1] + size
        else:
            self.size = [1] + [size] * 3

    def __call__(self, data_3d):
        patch = torch.zeros(self.size)

        init_x = random.randint(0, data_3d.shape[1] - self.size[0])
        init_y = random.randint(0, data_3d.shape[2] - self.size[1])
        init_z = random.randint(0, data_3d.shape[3] - self.size[2])

        patch = data_3d[:, init_x:init_x + self.size[0], init_y:init_y + self.size[1], init_z:init_z + self.size[2]]

        return patch


class ToTensor(object):
    """Convert 3D ndarrays in sample to Tensors."""

    def __init__(self, add_dim=True):
        self.add_dim = add_dim

    def __call__(self, image):
        # # swap color axis because
        # # numpy image: H x W x C
        # # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        if self.add_dim:
            image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        return image


class RandomRotate(object):
    '''
    Class that performs Random rotation
    -------------------------------------------------------------------------------------

    -------------------------------------------------------------------------------------
    '''

    def __init__(self):
        pass

    def __call__(self, x):
        step = torch.randint(0, 4, (1, ), device=x.device)[0]
        x = torch.rot90(x, step, [1, 2])
        step = torch.randint(0, 4, (1, ), device=x.device)[0]
        x = torch.rot90(x, step, [2, 3])
        step = torch.randint(0, 4, (1, ), device=x.device)[0]
        x = torch.rot90(x, step, [1, 3])
        return x


class Rotate(object):
    '''
    Class that performs rotation
    -------------------------------------------------------------------------------------

    -------------------------------------------------------------------------------------
    '''

    def __init__(self):
        pass

    def __call__(self, x, steps=[0, 0, 0]):
        x = torch.rot90(x, steps[0], [1, 2])
        x = torch.rot90(x, steps[1], [2, 3])
        x = torch.rot90(x, steps[2], [1, 3])
        return x


def get_index(data_3d, size=64):
    """generate index for 3D array patching

    Args:
        data_3d (numpy array or torch tensor): input data
        size (int, optional): patch size. Defaults to 64.

    Returns:
        int tuple: index
    """
    if isinstance(size, (list)):
        size = size
    else:
        size = [size] * 3
    init_x = random.randint(0, data_3d.shape[0] - size[0])
    init_y = random.randint(0, data_3d.shape[1] - size[1])
    init_z = random.randint(0, data_3d.shape[2] - size[2])
    return init_x, init_y, init_z
