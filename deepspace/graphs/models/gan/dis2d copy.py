import numpy as np
import torch
from torch import nn
from deepspace.graphs.models.autoencoder.ae2d import EncoderBlock


def fc_layer(in_features, out_features, activation=None, batchnorm=True):
    layers = [nn.Linear(in_features, out_features)]

    if activation is not None:
        layers += [activation]
    if batchnorm:
        layers += [nn.BatchNorm1d(out_features)]

    return nn.Sequential(*layers)


class Discriminator(nn.Module):
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

    def __init__(self, input_shape, encoder_sizes, fc_sizes, color_channels=1, latent_activation=None):
        super().__init__()
        conv_dims = list(zip([color_channels, *encoder_sizes], encoder_sizes))
        self.conv_encoder = nn.Sequential(*[EncoderBlock(*d) for d in conv_dims])

        # get the output shape, batch, channel, depth, height, width
        self.input_shape = (color_channels, *input_shape)
        with torch.no_grad():
            self.conv_encoder.eval()
            dummy_input = torch.Tensor(1, *self.input_shape)
            dummy_output = self.conv_encoder(dummy_input)

        self.intermediate_size = dummy_output.shape[1:]
        self.first_fc_size = np.prod(self.intermediate_size)

        fc_dims = list(zip([self.first_fc_size, *fc_sizes], fc_sizes))
        self.fc_encoder = nn.Sequential(
            *[fc_layer(*d, activation=nn.LeakyReLU()) for d in fc_dims[:-1]],
            fc_layer(*fc_dims[-1], activation=latent_activation, batchnorm=False))

    def encode(self, x):
        """
        >>> model = Residual3DAE((16, 64, 64), [4], [], color_channels=4, temporal_strides=[2])
        >>> x = torch.randn(2, 4, 16, 64, 64)
        >>> y = model.encode(x)
        >>> tuple(y.shape)
        (2, 8, 4096)
        """
        y = self.conv_encoder(x)
        y = y.view(-1, self.first_fc_size)
        y = self.fc_encoder(y)
        return y

    def forward(self, x):
        """
        >>> model = Discriminator3D((16, 64, 64), [10], [5], color_channels=4, temporal_strides=[1])
        >>> x = torch.randn(2, 4, 16, 64, 64)
        >>> y = model(x)
        >>> tuple(y.shape)
        (2, 4, 16, 64, 64)
        """
        return self.encode(x)


if __name__ == '__main__':
    model = Discriminator((64, 64), [32], [5], color_channels=1)
    x = torch.randn(64, 1, 64, 64, 64)
    y = model.encode(x)
    print(y.shape)
