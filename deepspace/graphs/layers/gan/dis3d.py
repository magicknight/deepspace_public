"""
3D GAN discriminator model
"""
from torch import nn


class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_stride=2):
        super().__init__()

        self.residual_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1,
                      stride=(temporal_stride, 2, 2)),
            nn.BatchNorm3d(out_channels))

        self.conv_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3,
                      stride=(temporal_stride, 2, 2), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm3d(out_channels))

    def forward(self, x):
        return self.residual_path(x) + self.conv_path(x)
