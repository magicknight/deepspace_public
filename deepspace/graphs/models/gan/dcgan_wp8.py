"""
based on the paper: https://arxiv.org/pdf/1511.06434.pdf
date: 30 April 2018
"""
import torch
import torch.nn as nn

import json
from easydict import EasyDict as edict
from deepspace.graphs.weights_initializer import weights_init, xavier_weights
from commontools.setup import config, logger
"""
DCGAN generator model
"""


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lrelu = nn.LeakyReLU(config.deepspace.relu_slope, inplace=True)

        self.conv1 = nn.Conv2d(in_channels=config.deepspace.image_channels, out_channels=config.deepspace.num_filt_g, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels=config.deepspace.num_filt_g, out_channels=config.deepspace.num_filt_g * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm1e = nn.BatchNorm2d(config.deepspace.num_filt_g*2)

        self.conv3 = nn.Conv2d(in_channels=config.deepspace.num_filt_g*2, out_channels=config.deepspace.num_filt_g * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm2e = nn.BatchNorm2d(config.deepspace.num_filt_g*4)

        self.conv4 = nn.Conv2d(in_channels=config.deepspace.num_filt_g*4, out_channels=config.deepspace.num_filt_g*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm3e = nn.BatchNorm2d(config.deepspace.num_filt_g*8)

        self.conv5 = nn.Conv2d(in_channels=config.deepspace.num_filt_g*8, out_channels=config.deepspace.latent_space_size, kernel_size=4, stride=1, padding=0, bias=False)
        self.batch_norm4e = nn.BatchNorm2d(config.deepspace.latent_space_size)

        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(in_channels=config.deepspace.latent_space_size, out_channels=config.deepspace.num_filt_g * 8, kernel_size=4, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(config.deepspace.num_filt_g*8)

        self.deconv2 = nn.ConvTranspose2d(in_channels=config.deepspace.num_filt_g * 8, out_channels=config.deepspace.num_filt_g * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(config.deepspace.num_filt_g*4)

        self.deconv3 = nn.ConvTranspose2d(in_channels=config.deepspace.num_filt_g * 4, out_channels=config.deepspace.num_filt_g * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(config.deepspace.num_filt_g*2)

        self.deconv4 = nn.ConvTranspose2d(in_channels=config.deepspace.num_filt_g * 2, out_channels=config.deepspace.num_filt_g, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(config.deepspace.num_filt_g)

        self.deconv5 = nn.ConvTranspose2d(in_channels=config.deepspace.num_filt_g, out_channels=config.deepspace.image_channels, kernel_size=4, stride=2, padding=1, bias=False)

        self.out = nn.Tanh()

        self.apply(weights_init)

    def forward(self, x):
        out = self.conv1(x)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.batch_norm1e(out)
        out = self.lrelu(out)

        out = self.conv3(out)
        out = self.batch_norm2e(out)
        out = self.lrelu(out)

        out = self.conv4(out)
        out = self.batch_norm3e(out)
        out = self.lrelu(out)

        out = self.conv5(out)
        out = self.batch_norm4e(out)
        out = self.relu(out)

        out = self.deconv1(out)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)

        out = self.deconv3(out)
        out = self.batch_norm3(out)
        out = self.relu(out)

        out = self.deconv4(out)
        out = self.batch_norm4(out)
        out = self.relu(out)

        out = self.deconv5(out)

        out = self.out(out)

        return out


"""
DCGAN discriminator model
"""


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.LeakyReLU(config.deepspace.relu_slope, inplace=True)

        self.conv1 = nn.Conv2d(in_channels=config.deepspace.image_channels, out_channels=config.deepspace.num_filt_d, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels=config.deepspace.num_filt_d, out_channels=config.deepspace.num_filt_d*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(config.deepspace.num_filt_d*2)

        self.conv3 = nn.Conv2d(in_channels=config.deepspace.num_filt_d*2, out_channels=config.deepspace.num_filt_d*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(config.deepspace.num_filt_d*4)

        self.conv4 = nn.Conv2d(in_channels=config.deepspace.num_filt_d*4, out_channels=config.deepspace.num_filt_d*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(config.deepspace.num_filt_d*8)

        self.conv5 = nn.Conv2d(in_channels=config.deepspace.num_filt_d*8, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear((config.deepspace.image_resolution // (2 ** 5)) ** 2, 1)
        self.out = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.batch_norm2(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.batch_norm3(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.out(out)

        return out.view(-1, 1).squeeze(1)


"""
net testing
"""


def main():
    config = json.load(open('configs/json/dcgan_exp_0.json'))
    config = edict(config)

    inp = torch.autograd.Variable(torch.randn(config.batch_size, config.deepspace.image_channels, config.image_size, config.image_size))
    print(inp.shape)
    netD = Discriminator(config)
    out = netD(inp)
    print(out)

    inp = torch.autograd.Variable(torch.randn(config.batch_size, config.g_input_size, 1, 1))
    print(inp.shape)
    netG = Generator(config)
    out = netG(inp)
    print(out.shape)


if __name__ == '__main__':
    main()
