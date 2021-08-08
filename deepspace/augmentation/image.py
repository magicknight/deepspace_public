import random
import math
import numpy as np
from torchvision import transforms
import torch


class CornerErasing(object):
    '''
    Class that performs Random Erasing on one of the 4 corners
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed on each corner
    value: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, size, probability=[0.25, 0.25, 0.25, 0.25], value=0):
        self.size = size
        self.probability = probability
        self.value = value

    def __call__(self, img):
        pick = random.uniform(0, 1)
        if pick > 0 and pick < self.probability[0]:
            return transforms.functional.erase(img, 0, 0, self.size, self.size, self.value)
        elif pick > self.probability[0] and pick < np.sum(self.probability[0:2]):
            return transforms.functional.erase(img, 0, self.size, self.size, self.size, self.value)
        elif pick > np.sum(self.probability[0:2]) and pick < np.sum(self.probability[0:3]):
            return transforms.functional.erase(img, self.size, 0, self.size, self.size, self.value)
        elif pick > np.sum(self.probability[0:3]) and pick < np.sum(self.probability[0:4]):
            return transforms.functional.erase(img, self.size, self.size, self.size, self.size, self.value)
        else:
            return None


class RandomBreak(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, value=0, return_normal=True):
        self.probability = probability
        self.value = value
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.return_normal = return_normal

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            if self.return_normal:
                return img, img
            else:
                return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                img_bk = transforms.functional.erase(img, x1, y1, w, h, self.value)
                if self.return_normal:
                    return img_bk, img
                else:
                    return img_bk
        if self.return_normal:
            return img, img
        else:
            return img


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img


def RandomRotation(X, orientation=None):
    # generate random number between 0 and n_rot to represent the rotation
    if orientation is None:
        orientation = np.random.randint(0, 4)

    if orientation == 0:  # do nothing
        pass
    elif orientation == 1:
        X = X.rot90(-1, [1, 2])
    elif orientation == 2:
        X = X.rot90(-1, [1, 2]).rot90(-1, [1, 2])
    elif orientation == 3:
        X = X.rot90(1, [1, 2])

    return X, orientation


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
