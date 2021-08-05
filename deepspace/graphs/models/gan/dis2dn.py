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
Date: 2021-07-08 07:31:14
LastEditors: Zhihua Liang
LastEditTime: 2021-08-04 00:04:25
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/models/gan/dis2dn.py
'''

import torch
from torch import nn
from deepspace.graphs.models.autoencoder.ae2dn import EncoderBlock


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

    def __init__(self, input_shape, encoder_sizes, fc_sizes, temporal_strides, color_channels=1, latent_activation=None):
        super().__init__()
        conv_dims = list(zip([color_channels, *encoder_sizes], encoder_sizes))
        temporal_strides = list(temporal_strides)
        assert len(temporal_strides) == len(conv_dims)
        self.conv_encoder = nn.Sequential(
            *[EncoderBlock(d_in, d_out, temporal_stride=ts)
              for (d_in, d_out), ts in zip(conv_dims, temporal_strides)]
        )

        # get the output shape, batch, channel, depth, height, width
        self.input_shape = (color_channels, *input_shape)
        with torch.no_grad():
            self.conv_encoder.eval()
            dummy_input = torch.Tensor(1, *self.input_shape)
            dummy_output = self.conv_encoder(dummy_input)
        _, c, h, w = dummy_output.shape
        self.intermediate_size = (c, h, w)
        self.first_fc_size = c * h * w

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
    from torchinfo import summary
    model = Discriminator(
        input_shape=(768, 768),
        encoder_sizes=[2, 4, 8, 16, 32, 64, 96, 128],
        fc_sizes=[64, 16, 1],
        temporal_strides=[2, 2, 2, 2, 2, 2, 2, 2],
        color_channels=1
    )
    summary(model, (2, 1, 768, 768),  depth=6)
    # x = torch.randn(2, 1, 784, 784)
    # y = model.encode(x)
    # print(y.shape)
