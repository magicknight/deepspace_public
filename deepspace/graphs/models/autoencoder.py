import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from deepspace.graphs.layers.gdn import GDN


class AutoEncoder(nn.Module):
    # https://arxiv.org/pdf/1611.01704.pdf
    # A simplfied version without quantization
    def __init__(self, C=128, M=128, in_chan=3, out_chan=3):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(C=C, M=M, in_chan=in_chan)
        self.decoder = Decoder(C=C, M=M, out_chan=out_chan)

    def forward(self, x, **kargs):
        code = self.encoder(x)
        out = self.decoder(code)
        return out


class Encoder(nn.Module):
    """ Encoder
    """

    def __init__(self, C=32, M=128, in_chan=3):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),
            GDN(M),

            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),
            GDN(M),

            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),
            GDN(M),

            nn.Conv2d(in_channels=M, out_channels=C, kernel_size=5, stride=2, padding=2, bias=False)
        )

    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Module):
    """ Decoder
    """

    def __init__(self, C=32, M=128, out_chan=3):
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=C, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            GDN(M, inverse=True),

            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            GDN(M, inverse=True),

            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            GDN(M, inverse=True),

            nn.ConvTranspose2d(in_channels=M, out_channels=out_chan, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
        )

    def forward(self, q):
        return torch.sigmoid(self.dec(q))


class AnomalyAE(nn.Module):
    """autoencoder for anomaly images

    Args:
        nn (torch nn): torch nn
    """

    def __init__(self, C=96, M=48, in_chan=1, out_chan=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=M, kernel_size=(11, 11), stride=(1, 1), padding=5)
        self.bn1 = nn.BatchNorm2d(M)

        self.conv2 = nn.Conv2d(in_channels=M, out_channels=M, kernel_size=(9, 9), stride=(2, 2), padding=4)
        self.bn2 = nn.BatchNorm2d(M)

        self.conv3 = nn.Conv2d(in_channels=M, out_channels=M, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn3 = nn.BatchNorm2d(M)

        self.conv4 = nn.Conv2d(in_channels=M, out_channels=M, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.bn4 = nn.BatchNorm2d(M)

        self.conv5 = nn.Conv2d(in_channels=M, out_channels=M, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn5 = nn.BatchNorm2d(M)

        self.conv_tr1 = nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1)
        self.bn_tr1 = nn.BatchNorm2d(M)

        self.conv_tr2 = nn.ConvTranspose2d(in_channels=C, out_channels=M, kernel_size=(7, 7), stride=(2, 2), padding=3, output_padding=1)
        self.bn_tr2 = nn.BatchNorm2d(M)

        self.conv_tr3 = nn.ConvTranspose2d(in_channels=C, out_channels=M, kernel_size=(9, 9), stride=(2, 2), padding=4, output_padding=1)
        self.bn_tr3 = nn.BatchNorm2d(M)

        self.conv_tr4 = nn.ConvTranspose2d(in_channels=C, out_channels=M, kernel_size=(11, 11), stride=(2, 2), padding=5, output_padding=1)
        self.bn_tr4 = nn.BatchNorm2d(M)

        self.conv_output = nn.Conv2d(in_channels=C, ut_channels=out_chan, kernel_size=(1, 1), stride=(1, 1))
        self.bn_output = nn.BatchNorm2d(1)

    def forward(self, x):
        slope = 0.2
        x = F.leaky_relu((self.bn1(self.conv1(x))), slope)
        x1 = F.leaky_relu((self.bn2(self.conv2(x))), slope)
        x2 = F.leaky_relu((self.bn3(self.conv3(x1))), slope)
        x3 = F.leaky_relu((self.bn4(self.conv4(x2))), slope)
        x4 = F.leaky_relu((self.bn5(self.conv5(x3))), slope)

        x5 = F.leaky_relu(self.bn_tr1(self.conv_tr1(x4)), slope)
        x6 = F.leaky_relu(self.bn_tr2(
            self.conv_tr2(torch.cat([x5, x3], 1))), slope)
        x7 = F.leaky_relu(self.bn_tr3(
            self.conv_tr3(torch.cat([x6, x2], 1))), slope)
        x8 = F.leaky_relu(self.bn_tr4(
            self.conv_tr4(torch.cat([x7, x1], 1))), slope)

        output = F.leaky_relu(self.bn_output(
            self.conv_output(torch.cat([x8, x], 1))), slope)
        return output


if __name__ == "__main__":
    x = torch.rand([16, 1, 512, 512])
    model = AnomalyAE()
    y = model(x)
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)
