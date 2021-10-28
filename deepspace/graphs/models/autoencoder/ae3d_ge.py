import torch
from torch import nn
from einops import rearrange

# __all__ = ['Residual3DAE']
# __all__ = ['Residual3DAE']


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, temporal_stride=2):
        super().__init__()

        self.residual_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(temporal_stride, temporal_stride, temporal_stride)),
            nn.BatchNorm3d(out_channels)
        )

        self.conv_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=(temporal_stride, temporal_stride, temporal_stride), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels))

    def forward(self, x):
        """
        >>> x = torch.randn(1, 3, 8, 32, 32)
        >>> block = EncoderBlock(3, 10)
        >>> tuple(block(x).shape)
        (1, 10, 4, 16, 16)
        """
        # return self.residual_path(x) + self.conv_path(x)
        return self.conv_path(x)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, temporal_stride=2):
        super().__init__()

        self.residual_path = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=(temporal_stride, temporal_stride, temporal_stride), output_padding=(temporal_stride - 1, temporal_stride - 1, temporal_stride - 1)),
            nn.BatchNorm3d(out_channels))

        self.conv_path = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=(temporal_stride, 2, 2), padding=1, output_padding=(temporal_stride - 1, temporal_stride - 1, temporal_stride - 1)),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        """
        >>> x = torch.randn(1, 3, 4, 16, 16)
        >>> block = DecoderBlock(3, 10)
        >>> tuple(block(x).shape)
        (1, 10, 8, 32, 32)
        """
        # return self.residual_path(x) + self.conv_path(x)
        return self.conv_path(x)


class ParallelBatchNorm1d(nn.BatchNorm1d):

    def forward(self, x):
        x_reshaped = x.view(-1, x.shape[1])
        y_reshaped = super().forward(x_reshaped)
        y = y_reshaped.view(*x.shape)
        return y

    @staticmethod
    def _get_name():
        return 'ParallelBatchNorm1d'


def fc_layer(in_features, out_features, activation=None, batchnorm=True):
    layers = [nn.Linear(in_features, out_features)]
    if batchnorm:
        layers += [ParallelBatchNorm1d(out_features)]
    if activation is not None:
        layers += [activation]

    return nn.Sequential(*layers)


class Residual3DAE(nn.Module):
    """   
    print('Setting up models')
    encoder = novelly.Residual3DAE(
        input_shape=(time_steps, height, width),
        encoder_sizes=[8, 16, 32, 64, 64],
        decoder_sizes=[64, 32, 16, 8, 8],
        temporal_strides=[2, 2, 1, 1, 1],
        fc_sizes=[512, 64],
        color_channels=3,
        latent_activation=nn.Sigmoid())

    Args:
        nn (nn): torch.nn
    """

    def __init__(self, input_shape, encoder_sizes, fc_sizes, *, temporal_strides,
                 decoder_sizes=None, color_channels=1, latent_activation=None):
        super().__init__()

        decoder_sizes = decoder_sizes if decoder_sizes is not None else list(reversed(encoder_sizes))

        conv_dims = list(zip([color_channels, *encoder_sizes], encoder_sizes))
        temporal_strides = list(temporal_strides)
        assert len(temporal_strides) == len(conv_dims)
        self.conv_encoder = nn.Sequential(
            *[EncoderBlock(d_in, d_out, temporal_stride=ts)
              for (d_in, d_out), ts in zip(conv_dims, temporal_strides)]
        )

        self.input_shape = (color_channels, *input_shape)
        with torch.no_grad():
            self.conv_encoder.eval()
            dummy_input = torch.Tensor(1, *self.input_shape)
            dummy_output = self.conv_encoder(dummy_input)
        _, c, t, h, w = dummy_output.shape
        self.intermediate_size = (c, t, h, w)
        self.first_fc_size = c * t * h * w

        fc_dims = list(zip([self.first_fc_size, *fc_sizes], fc_sizes))
        self.fc_encoder = nn.Sequential(
            *[fc_layer(*d, activation=nn.GELU()) for d in fc_dims[:-1]],
            fc_layer(*fc_dims[-1], activation=latent_activation, batchnorm=False))

        self.fc_decoder = nn.Sequential(
            *[fc_layer(d_out, d_in, activation=nn.GELU())
              for d_in, d_out in reversed(fc_dims)])
        conv_dims = list(zip(decoder_sizes, [*(decoder_sizes[1:]), color_channels]))
        self.conv_decoder = nn.Sequential(
            *[DecoderBlock(d_in, d_out, temporal_stride=ts) for (d_in, d_out), ts in zip(conv_dims, reversed(temporal_strides))], nn.Sigmoid()
        )

    def encode(self, x):
        """
        >>> model = Residual3DAE((16, 64, 64), [4], [], color_channels=4, temporal_strides=[2])
        >>> x = torch.randn(2, 4, 16, 64, 64)
        >>> y = model.encode(x)
        >>> tuple(y.shape)
        (2, 8, 4096)
        """
        y = self.conv_encoder(x)
        # Group together CWH indices, but keep time index separate
        # y = y.permute(0, 2, 1, 3, 4).contiguous().view(-1, *self.first_fc_size)
        # y = y.contiguous().view(-1, *self.first_fc_size)
        # y = y.view(-1, *self.first_fc_size)
        y = rearrange(y, 'b c t w h -> b (c t w h)')
        y = self.fc_encoder(y)
        return y

    def decode(self, x):
        y = self.fc_decoder(x)
        # y = y.view(-1, *self.intermediate_shape).permute(0, 2, 1, 3, 4)
        # y = y.view(-1, *self.intermediate_shape)
        y = rearrange(y, 'b (c t w h) -> b c t w h', c=self.intermediate_size[0], t=self.intermediate_size[1], w=self.intermediate_size[2], h=self.intermediate_size[3])
        y = self.conv_decoder(y)
        return y

    def forward(self, x):
        """
        >>> model = Residual3DAE((16, 64, 64), [10], [5], color_channels=4, temporal_strides=[1])
        >>> x = torch.randn(2, 4, 16, 64, 64)
        >>> y = model(x)
        >>> tuple(y.shape)
        (2, 4, 16, 64, 64)
        """
        return self.decode(self.encode(x))


if __name__ == "__main__":
    from torchinfo import summary
    model = Residual3DAE(
        input_shape=(64, 64, 64),
        encoder_sizes=[4, 16, 64, 256],
        fc_sizes=[1024],
        temporal_strides=[2, 2, 2, 2],
        color_channels=1
    )
    summary(model, (2, 1, 64, 64, 64),  depth=6)
    # x = torch.randn(2, 1, 784, 784)
    # y = model(x)
    # print(y.shape)
