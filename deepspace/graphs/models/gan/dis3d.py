'''
 ┌─────────────────────────────────────────────────────────────┐
 │┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
 ││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
 │├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
 ││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
 │├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
 ││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
 │├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
 ││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
 │└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
 │      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
 │      └───┴─────┴───────────────────────┴─────┴───┘          │
 └─────────────────────────────────────────────────────────────┘

Description: 
Author: Zhihua Liang
Github: https://github.com/magicknight
Date: 2021-07-12 09:41:12
LastEditors: Zhihua Liang
LastEditTime: 2021-07-12 09:41:13
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/models/gan/dis3d.py
'''

import torch
from torch import nn
from deepspace.graphs.models.autoencoder.ae3d_ge import EncoderBlock


def fc_layer(in_features, out_features, activation=None, batchnorm=True):
    layers = [nn.Linear(in_features, out_features)]

    if batchnorm:
        layers += [nn.BatchNorm1d(out_features)]
    if activation is not None:
        layers += [activation]

    return nn.Sequential(*layers)


class Discriminator3D(nn.Module):
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

    def __init__(self, input_shape, encoder_sizes, fc_sizes, temporal_strides, color_channels=1, latent_activation=None):
        super().__init__()
        # from color_channels to the last encoder_sizes
        conv_dims = list(zip([color_channels, *encoder_sizes], encoder_sizes))
        temporal_strides = list(temporal_strides)
        assert len(temporal_strides) == len(conv_dims)

        self.conv_encoder = nn.Sequential(*[EncoderBlock(d_in, d_out, temporal_stride=ts) for (d_in, d_out), ts in zip(conv_dims, temporal_strides)])

        # get the output shape, batch, channel, depth, height, width
        self.input_shape = (color_channels, *input_shape)
        with torch.no_grad():
            self.conv_encoder.eval()
            dummy_input = torch.Tensor(1, *self.input_shape)
            dummy_output = self.conv_encoder(dummy_input)
        _, c, t, h, w = dummy_output.shape
        self.intermediate_shape = (c, t, h, w)
        self.first_fc_size = c * t * h * w
        # fully connected layer input size to fc_sizes last
        fc_dims = list(zip([self.first_fc_size, *fc_sizes], fc_sizes))
        if fc_dims:
            self.fc_encoder = nn.Sequential(
                *[fc_layer(d_in, d_out, activation=nn.GELU()) for d_in, d_out in fc_dims[:-1]],
                fc_layer(*fc_dims[-1], activation=latent_activation, batchnorm=False))
        else:
            self.fc_encoder = nn.Sequential()

    def encode(self, x):
        """
        >>> model = Residual3DAE((16, 64, 64), [4], [], color_channels=4, temporal_strides=[2])
        >>> x = torch.randn(2, 4, 16, 64, 64)
        >>> y = model.encode(x)
        >>> tuple(y.shape)
        (2, 8, 4096)
        """
        y = self.conv_encoder(x)
        # y = y.contiguous().view(-1, self.first_fc_size)
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
    model = Discriminator3D((64, 64, 64), [2, 1], [16, 1], [2, 1], color_channels=1)
    x = torch.randn(64, 1, 64, 64, 64)
    y = model.encode(x)
    print(y.shape)
