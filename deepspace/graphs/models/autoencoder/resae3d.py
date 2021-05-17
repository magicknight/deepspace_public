
"""
https://github.com/dsuess/autoregressive-novelty-detection
"""
import torch
from torch import nn
from deepspace.graphs.layers.ae.encoder3d import EncoderBlock
from deepspace.graphs.layers.ae.decoder3d import DecoderBlock
from deepspace.graphs.layers.ae.misc import fc_layer


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
        decoder_sizes = decoder_sizes if decoder_sizes is not None \
            else list(reversed(encoder_sizes))

        conv_dims = list(zip([color_channels, *encoder_sizes], encoder_sizes))
        temporal_strides = list(temporal_strides)
        assert len(temporal_strides) == len(conv_dims)

        self.conv_encoder = nn.Sequential(
            *[EncoderBlock(d_in, d_out, temporal_stride=ts)
              for (d_in, d_out), ts in zip(conv_dims, temporal_strides)])

        self.input_shape = (color_channels, *input_shape)
        with torch.no_grad():
            self.conv_encoder.eval()
            dummy_input = torch.Tensor(1, *self.input_shape)
            dummy_output = self.conv_encoder(dummy_input)
        _, c, t, h, w = dummy_output.shape
        self.intermediate_shape = (c, t, h, w)
        self.first_fc_size = (c, t * h * w)
        fc_dims = list(zip([self.first_fc_size[1], *fc_sizes], fc_sizes))
        if fc_dims:
            self.fc_encoder = nn.Sequential(
                *[fc_layer(d_in, d_out, activation=nn.LeakyReLU())
                  for d_in, d_out in fc_dims[:-1]],
                fc_layer(*fc_dims[-1], activation=latent_activation,
                         batchnorm=False))
        else:
            self.fc_encoder = nn.Sequential()

        self.fc_decoder = nn.Sequential(
            *[fc_layer(d_out, d_in, activation=nn.LeakyReLU())
              for d_in, d_out in reversed(fc_dims)])
        conv_dims = list(zip(decoder_sizes, [*decoder_sizes[1:], color_channels]))
        self.conv_decoder = nn.Sequential(*[DecoderBlock(d_in, d_out, temporal_stride=ts) for (d_in, d_out), ts in zip(conv_dims, reversed(temporal_strides))], nn.Sigmoid())

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
        y = y.contiguous().view(-1, *self.first_fc_size)
        y = self.fc_encoder(y)
        return y

    def decode(self, x):
        y = self.fc_decoder(x)
        # y = y.view(-1, *self.intermediate_shape).permute(0, 2, 1, 3, 4)
        y = y.view(-1, *self.intermediate_shape)
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
