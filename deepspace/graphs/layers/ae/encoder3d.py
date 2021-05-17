from torch import nn
from torch.nn import Conv3d


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, temporal_stride=2):
        super().__init__()

        self.residual_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1,
                      stride=(temporal_stride, 2, 2)),
            nn.BatchNorm3d(out_channels))

        self.conv_path = nn.Sequential(
            Conv3d(in_channels, out_channels, kernel_size=3,
                   stride=(temporal_stride, 2, 2), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(out_channels),
            Conv3d(out_channels, out_channels, kernel_size=3, stride=1,
                   padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(out_channels),
            Conv3d(out_channels, out_channels, kernel_size=3, stride=1,
                   padding=1),
            nn.BatchNorm3d(out_channels))

    def forward(self, x):
        """
        >>> x = torch.randn(1, 3, 8, 32, 32)
        >>> block = EncoderBlock(3, 10)
        >>> tuple(block(x).shape)
        (1, 10, 4, 16, 16)
        """
        return self.residual_path(x) + self.conv_path(x)
