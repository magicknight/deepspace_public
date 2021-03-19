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

operation_canditates = {
    '00': lambda filter1, filter2, stride, dilation: SeparableConv2d(filter1, filter2, 3, stride, dilation),
    '01': lambda filter1, filter2, stride, dilation: SepConv(filter1, filter2, 3, stride, 1),
    '02': lambda filter1, filter2, stride, dilation: SepConv(filter1, filter2, 5, stride, 2),
    '03': lambda filter1, filter2, stride, dilation: Identity(),
}


def get_pf_list():
    pf1 = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0]]).astype('float32')

    pf2 = np.array([[0, 0, 0],
                    [0, -1, 1],
                    [0, 0, 0]]).astype('float32')

    pf3 = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]]).astype('float32')

    return [torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone(),
            torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone(),
            torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone()
            ]


def constrained_weights(weights):
    weights = weights.permute(2, 3, 0, 1)
    # Scale by 10k to avoid numerical issues while normalizing
    weights = weights * 10000

    # Set central values to zero to exlude them from the normalization step
    weights[2, 2, :, :] = 0

    # Pass the weights
    filter_1 = weights[:, :, 0, 0]
    filter_2 = weights[:, :, 0, 1]
    filter_3 = weights[:, :, 0, 2]

    # Normalize the weights for each filter.
    # Sum in the 3rd dimension, which contains 25 numbers.
    filter_1 = filter_1.reshape(1, 1, 1, 25)
    filter_1 = filter_1 / filter_1.sum(3).reshape(1, 1, 1, 1)
    filter_1[0, 0, 0, 12] = -1

    filter_2 = filter_2.reshape(1, 1, 1, 25)
    filter_2 = filter_2 / filter_2.sum(3).reshape(1, 1, 1, 1)
    filter_2[0, 0, 0, 12] = -1

    filter_3 = filter_3.reshape(1, 1, 1, 25)
    filter_3 = filter_3 / filter_3.sum(3).reshape(1, 1, 1, 1)
    filter_3[0, 0, 0, 12] = -1

    # Prints are for debug reasons.
    # The sums of all filter weights for a specific filter
    # should be very close to zero.
    # print(filter_1)
    # print(filter_2)
    # print(filter_3)
    # print(filter_1.sum(3).reshape(1,1,1,1))
    # print(filter_2.sum(3).reshape(1,1,1,1))
    # print(filter_3.sum(3).reshape(1,1,1,1))

    # Reshape to original size.
    filter_1 = filter_1.reshape(1, 1, 5, 5)
    filter_2 = filter_2.reshape(1, 1, 5, 5)
    filter_3 = filter_3.reshape(1, 1, 5, 5)

    # Pass the weights back to the original matrix and return.
    weights[:, :, 0, 0] = filter_1
    weights[:, :, 0, 1] = filter_2
    weights[:, :, 0, 2] = filter_3

    weights = weights.permute(2, 3, 0, 1)
    return weights


class CustomizedConv(nn.Module):
    def __init__(self, channels=1, choice='similarity'):
        super(CustomizedConv, self).__init__()
        self.channels = channels
        self.choice = choice
        kernel = [[0.03598, 0.03735, 0.03997, 0.03713, 0.03579],
                  [0.03682, 0.03954, 0.04446, 0.03933, 0.03673],
                  [0.03864, 0.04242, 0.07146, 0.04239, 0.03859],
                  [0.03679, 0.03936, 0.04443, 0.03950, 0.03679],
                  [0.03590, 0.03720, 0.04003, 0.03738, 0.03601]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.kernel = nn.modules.utils._pair(3)
        self.stride = nn.modules.utils._pair(1)
        self.padding = nn.modules.utils._quadruple(0)
        self.same = False

    def __call__(self, x):
        if self.choice == 'median':
            x = F.pad(x, self._padding(x), mode='reflect')
            x = x.unfold(2, self.kernel[0], self.stride[0]).unfold(3, self.kernel[1], self.stride[1])
            x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        else:
            x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding


class Global_Local_Attention(nn.Module):
    def __init__(self):
        super(Global_Local_Attention, self).__init__()
        self.local_att = CustomizedConv(256, choice='similarity')

    def forward(self, x):
        tmp = x
        former = tmp.permute(0, 2, 3, 1).view(tmp.size(0), tmp.size(2) * tmp.size(3), -1)
        num_0 = torch.einsum('bik,bjk->bij', [former, former])
        norm_former = torch.einsum("bij,bij->bi", [former, former])
        den_0 = torch.sqrt(torch.einsum('bi,bj->bij', [norm_former, norm_former]))
        cosine = num_0 / den_0

        F_local = self.local_att(x.clone())

        top_T = 15  # The default maximum value of T is 15
        cosine_max, indexes = torch.topk(cosine, top_T, dim=2)
        dy_T = top_T
        for t in range(top_T):
            if torch.mean(cosine_max[:, :, t]) >= 0.5:
                dy_T = t
        dy_T = max(2, dy_T)

        mask = torch.ones(tmp.size(0), tmp.size(2) * tmp.size(3)).cuda()
        mask_index = (mask == 1).nonzero()[:, 1].view(tmp.size(0), -1)
        idx_b = torch.arange(tmp.size(0)).long().unsqueeze(1).expand(tmp.size(0), mask_index.size(1))

        rtn = tmp.clone().permute(0, 2, 3, 1).view(tmp.size(0), tmp.size(2) * tmp.size(3), -1)
        for t in range(1, dy_T):
            mask_index_top = (mask == 1).nonzero()[:, 1].view(tmp.size(0), -1).gather(1, indexes[:, :, t])
            ind_1st_top = torch.zeros(tmp.size(0), tmp.size(2) * tmp.size(3), tmp.size(2) * tmp.size(3)).cuda()
            ind_1st_top[(idx_b, mask_index, mask_index_top)] = 1
            rtn += torch.bmm(ind_1st_top, former)
        rtn = rtn / dy_T
        F_global = rtn.permute(0, 2, 1).view(tmp.shape)
        # The following line maybe useful when the location of Attention() in the network is changed.
        # F_global = nn.UpsamplingNearest2d(size=(x.shape[2], x.shape[3]))(rtn.float())
        x = torch.cat([x, F_global, F_local], dim=1)
        return x


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=False),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None,
                 start_with_relu=True, grow_first=True, genotype=None):
        super(Block, self).__init__()
        if not genotype:
            genotype = ['03', '03', '03']

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU()
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(operation_canditates[genotype[i]](filters, filters, 1, dilation))
            rep.append(BatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        rep.append(self.relu)
        rep.append(operation_canditates[genotype[2]](filters, filters, stride, 1))
        rep.append(BatchNorm(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x
