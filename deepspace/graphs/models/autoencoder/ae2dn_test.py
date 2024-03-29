"""
This is a test version of the autoencoder.
This version returns the output of each encoder layer.
"""
from pyexpat import features
import torch
from torch import nn
from einops import rearrange


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_stride=2):
        super().__init__()

        self.residual_path = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(temporal_stride, temporal_stride),
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.conv_path = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=(temporal_stride, temporal_stride),
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """
        >>> x = torch.randn(1, 3, 32, 32)
        >>> block = EncoderBlock(3, 10)
        >>> tuple(block(x).shape)
        (1, 10, 16, 16)
        """
        # return self.residual_path(x) + self.conv_path(x)
        return self.conv_path(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_stride=2):
        super().__init__()

        self.residual_path = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(temporal_stride, temporal_stride),
                output_padding=(temporal_stride - 1, temporal_stride - 1),
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.conv_path = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=(temporal_stride, temporal_stride),
                padding=1,
                output_padding=(temporal_stride - 1, temporal_stride - 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.Tanh()
        )

    def forward(self, x):
        """
        >>> x = torch.randn(1, 3, 16, 16)
        >>> block = DecoderBlock(3, 10)
        >>> tuple(block(x).shape)
        (1, 10, 32, 32)
        """
        # return self.residual_path(x) + self.conv_path(x)
        return self.conv_path(x)


def fc_layer(in_features, out_features, activation=None, batchnorm=True):
    layers = [nn.Linear(in_features, out_features)]
    if batchnorm:
        layers += [nn.BatchNorm1d(out_features)]
    if activation is not None:
        layers += [activation]
    return nn.Sequential(*layers)


class ResidualAE(nn.Module):
    def __init__(self, input_shape, encoder_sizes, fc_sizes, *, temporal_strides, color_channels=1, latent_activation=None, decoder_sizes=None):
        super().__init__()

        decoder_sizes = list(decoder_sizes) if decoder_sizes is not None else list(reversed(encoder_sizes))

        conv_dims = list(zip([color_channels, *encoder_sizes], encoder_sizes))
        temporal_strides = list(temporal_strides)
        assert len(temporal_strides) == len(conv_dims)
        self.conv_encoder = nn.Sequential(*[EncoderBlock(d_in, d_out, temporal_stride=ts) for (d_in, d_out), ts in zip(conv_dims, temporal_strides)])

        self.input_shape = (color_channels, *input_shape)
        with torch.no_grad():
            self.conv_encoder.eval()
            dummy_input = torch.Tensor(1, *self.input_shape)
            dummy_output = self.conv_encoder(dummy_input)
        _, c, w, h = dummy_output.shape
        self.intermediate_size = (c, w, h)
        self.first_fc_size = c * w * h

        fc_dims = list(zip([self.first_fc_size, *fc_sizes], fc_sizes))
        self.fc_encoder = nn.Sequential(*[fc_layer(*d, activation=nn.GELU()) for d in fc_dims[:-1]], fc_layer(*fc_dims[-1], activation=latent_activation, batchnorm=False))
        # self.embedding_dim = fc_sizes[-1]

        self.fc_decoder = nn.Sequential(*[fc_layer(d_out, d_in, activation=nn.GELU()) for d_in, d_out in reversed(fc_dims)])
        conv_dims = list(zip(decoder_sizes, [*(decoder_sizes[1:]), color_channels]))
        self.conv_decoder = nn.Sequential(*[DecoderBlock(d_in, d_out, temporal_stride=ts) for (d_in, d_out), ts in zip(conv_dims, reversed(temporal_strides))], nn.Sigmoid())

    def encode(self, x):
        features = []
        for layer in self.conv_encoder:
            x = layer(x)
            features.append(x)
        x = rearrange(x, "b c w h -> b (c w h)")
        x = self.fc_encoder(x)
        return x, features

    def decode(self, x):
        features = []
        x = self.fc_decoder(x)
        x = rearrange(
            x,
            "b (c w h) -> b c w h",
            c=self.intermediate_size[0],
            w=self.intermediate_size[1],
            h=self.intermediate_size[2],
        )
        # x = self.conv_decoder(x)
        for layer in self.conv_decoder:
            x = layer(x)
            features.append(x)
        return x, features

    def forward(self, x):
        """
        >>> model = ResidualAE((64, 64), [32], [5], color_channels=4)
        >>> x = torch.randn(2, 4, 64, 64)
        >>> y = model(x)
        >>> tuple(y.shape)
        (2, 4, 64, 64)
        """
        x, features_encode = self.encode(x)
        x, features_decode = self.decode(x)
        return x, features_encode, features_decode


if __name__ == "__main__":
    from torchinfo import summary

    model = ResidualAE(
        input_shape=(768, 768),
        encoder_sizes=[2, 4, 8, 16, 32, 64, 96, 128],
        fc_sizes=[1024],
        temporal_strides=[2, 2, 2, 2, 2, 2, 2, 2],
        color_channels=1,
    )
    summary(model, (2, 1, 768, 768), depth=6)
    # x = torch.randn(2, 1, 784, 784)
    # y = model(x)
    # print(y.shape)
