"""
Huber Loss for DQN model
"""
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 100*(1 - super(MS_SSIM_Loss, self).forward(img1, img2))


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100*(1 - super(SSIM_Loss, self).forward(img1, img2))
