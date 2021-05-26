# Fast and differentiable MS-SSIM and SSIM for pytorch

import warnings

import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.nn.functional import mse_loss

from deepspace.utils.data import make_masks_tensor
from commontools.setup import config, logger


class MaskLoss(MSELoss):
    """a mean square error loss function whhich calculate the MSE with the mask of the compared image.
    """

    def __init__(self):
        super().__init__()

    def forward(self, image_1, image_2, eps=1e-8) -> float:
        """compare two images in mask which presents the area of the non-zero elements.

        Args:
            image_1 (np array): image 1
            image_2 (np array): image 2

        Returns:
            float: return the percentage of the overlap area
        """
        image_2 = make_masks_tensor(image_2, config.swap.device)
        return super().forward(image_1, image_2)


class NeutrinoLoss(MSELoss):
    """a mean square error loss function whhich calculate the MSE with the mask of the compared image.
    """

    def __init__(self):
        super().__init__()

    def forward(self, p_1, p_2, eps=1e-8) -> float:
        """compare two momentum. mse((p1 - p2) / sqrt(p2))

        Args:
            p_1 (np array): momentum 1
            p_2 (np array): momentum 2

        Returns:
            float: return the percentage of the overlap area
        """
        # p_1 = torch.div(p_1, p_2 + eps)
        # p_2 = torch.div(p_2, p_2 + eps)
        return super().forward(p_1, p_2)
