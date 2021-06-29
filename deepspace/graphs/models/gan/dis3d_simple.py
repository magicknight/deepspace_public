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
Date: 2021-06-24 22:06:36
LastEditors: Zhihua Liang
LastEditTime: 2021-06-24 22:06:40
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/models/gan/dis3d_simple.py
'''
import torch


class Discriminator3D(torch.nn.Module):
    def __init__(self, cube_len, leak_value, bias):
        super(Discriminator3D, self).__init__()
        self.cube_len = cube_len[0]
        self.leak_value = leak_value
        self.bias = bias

        padd = (0, 0, 0)
        if self.cube_len == 32:
            padd = (1, 1, 1)

        self.f_dim = self.cube_len

        self.layer1 = self.conv_layer(1, self.f_dim, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer2 = self.conv_layer(self.f_dim, self.f_dim*2, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer3 = self.conv_layer(self.f_dim*2, self.f_dim*4, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer4 = self.conv_layer(self.f_dim*4, self.f_dim*8, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.f_dim*8, 1, kernel_size=4, stride=2, bias=self.bias, padding=padd),
            torch.nn.Sigmoid()
        )

        # self.layer5 = torch.nn.Sequential(
        #     torch.nn.Linear(256*2*2*2, 1),
        #     torch.nn.Sigmoid()
        # )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1, 1, 1), bias=False):
        # print(input_dim, output_dim, kernel_size, stride, bias, padding)
        layer = torch.nn.Sequential(
            torch.nn.Conv3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            torch.nn.LeakyReLU(self.leak_value, inplace=True)
        )
        return layer

    def forward(self, x):
        # out = torch.unsqueeze(x, dim=1)
        # out = x.view(-1, 1, self.cube_len, self.cube_len, self.cube_len)
        # print(out.size()) # torch.Size([32, 1, 32, 32, 32])
        out = self.layer1(x)
        # print(out.size())  # torch.Size([32, 32, 16, 16, 16])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([32, 64, 8, 8, 8])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([32, 128, 4, 4, 4])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([32, 256, 2, 2, 2])
        # out = out.view(-1, 256*2*2*2)
        # print (out.size())
        out = self.layer5(out)
        # print(out.size())  # torch.Size([32, 1, 1, 1, 1])
        out = torch.squeeze(out)
        return out
