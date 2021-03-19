import torch
from torch import nn


class Generator(nn.Module):

    # generator model
    def __init__(self, in_channels=1):
        super(Generator, self).__init__()

        self.t1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        # self.t6 = nn.Sequential(
        #     nn.Conv2d(512, 4000, kernel_size=(4, 4)),  # bottleneck
        #     nn.BatchNorm2d(4000),
        #     nn.ReLU()
        # )
        self.t8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.t9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.t10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.t11 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.t12 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.t14 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=in_channels, kernel_size=(4, 4), stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        x = self.t5(x)
        x = self.t6(x)
        x = self.t7(x)
        x = self.t8(x)
        x = self.t9(x)
        x = self.t10(x)
        x = self.t11(x)
        x = self.t12(x)
        x = self.t13(x)
        x = self.t14(x)
        return x  # output of generator


class Discriminator(nn.Module):

    # discriminator model
    def __init__(self,  in_channels=1):
        super(Discriminator, self).__init__()

        self.t1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # self.t7 = nn.Sequential(
        #     nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=(4, 4), stride=2, padding=1),
        #     nn.BatchNorm2d(4096),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        self.t8 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=(4, 4), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        x = self.t5(x)
        x = self.t6(x)
        # x = self.t7(x)
        x = self.t8(x)
        return x  # output of discriminator
