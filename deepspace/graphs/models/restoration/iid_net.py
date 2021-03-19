import os
import cv2
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score

from deepspace.graphs.layers.restoration.iid_net import Block, Global_Local_Attention, get_pf_list, CustomizedConv, constrained_weights


class IID_Net(nn.Module):
    def __init__(self):
        super(IID_Net, self).__init__()
        BatchNorm = nn.BatchNorm2d

        # The Enhancement Block
        self.normal_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.pf_conv = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1, bias=False)
        self.bayar_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1 = nn.Conv2d(15, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(64)

        # The Extraction Block
        # The code in 'genotype' means the best architecture selected from 1000 sampled candidate models.
        self.cell1 = Block(64, 128, reps=3, stride=2, BatchNorm=BatchNorm, start_with_relu=False, genotype=['01', '03', '00'])
        self.cell2 = Block(128, 256, reps=3, stride=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['02', '00', '00'])
        self.cell3 = Block(256, 256, reps=3, stride=1, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['03', '02', '00'])
        self.cell4 = Block(256, 256, reps=3, stride=1, dilation=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['01', '00', '01'])
        self.cell5 = Block(256, 256, reps=3, stride=1, dilation=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['00', '02', '00'])
        self.cell6 = Block(256, 256, reps=3, stride=1, dilation=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['00', '01', '00'])
        self.cell7 = Block(256, 256, reps=3, stride=1, dilation=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['02', '03', '02'])
        self.cell8 = Block(256, 256, reps=3, stride=1, dilation=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['03', '03', '00'])
        self.cell9 = Block(256, 256, reps=3, stride=1, dilation=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['02', '02', '00'])
        self.cell10 = Block(256, 256, reps=3, stride=1, dilation=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['00', '01', '03'])

        # The Decision Block
        self.att = Global_Local_Attention()
        self.decision = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256 * 3, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 1, 3, stride=1, padding=1),
        )

        self.pf_list = get_pf_list()
        self.reset_pf()
        self.median = CustomizedConv(choice='median')

    def forward(self, x):
        _, _, H, W = x.shape

        # The Enhancement Block
        self.bayar_conv.weight.data = constrained_weights(self.bayar_conv.weight.data)
        bayar_x = self.bayar_conv(x)
        normal_x = self.normal_conv(x)
        pf_x = self.pf_conv(x)
        x = torch.cat([normal_x, bayar_x, pf_x], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # The Extraction Block
        x = self.cell1(x)
        x = self.cell2(x)
        x = self.cell3(x)
        x = self.cell4(x)
        x = self.cell5(x)
        x = self.cell6(x)
        x = self.cell7(x)
        x = self.cell8(x)
        x = self.cell9(x)
        x = self.cell10(x)

        # The Decision Block
        x = self.att(x)
        x = self.decision(x)

        # The Median Filter is embedded here because we found that in some rare cases it would cause overflow
        # when calculating the loss functions, and this makes almost no difference.
        x = self.median(x)
        x = F.interpolate(x, (H, W))
        x = nn.Sigmoid()(x)
        return x

    def reset_pf(self):
        for idx, pf in enumerate(self.pf_list):
            self.pf_conv.weight.data[idx, :, :, :] = pf
