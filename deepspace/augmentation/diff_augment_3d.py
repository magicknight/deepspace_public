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
Date: 2021-06-28 17:59:41
LastEditors: Zhihua Liang
LastEditTime: 2021-06-28 17:59:41
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/transform/diff_augment_3d.py
'''

# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738
# https: // github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_pytorch.py


import torch
import torch.nn.functional as F


def DiffAugment(x, policy=''):
    if policy:
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3, 4], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y, shift_z = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5), int(x.size(4) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1, 1], device=x.device)
    translation_z = torch.randint(-shift_z, shift_z + 1, size=[x.size(0), 1, 1, 1], device=x.device)
    grid_batch, grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        torch.arange(x.size(4), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    grid_z = torch.clamp(grid_z + translation_z + 1, 0, x.size(4) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 4, 1).contiguous()[grid_batch, grid_x, grid_y, grid_z].permute(0, 4, 1, 2, 3).contiguous()
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5), int(x.size(4) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1, 1], device=x.device)
    offset_z = torch.randint(0, x.size(4) + (1 - cutout_size[2] % 2), size=[x.size(0), 1, 1, 1], device=x.device)
    grid_batch, grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[2], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    grid_z = torch.clamp(grid_z + offset_z - cutout_size[2] // 2, min=0, max=x.size(4) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), x.size(4), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y, grid_z] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
