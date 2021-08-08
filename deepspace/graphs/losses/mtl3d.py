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

Description: from github https://github.com/Sara-Ahmed/SiT/blob/main/losses.py
Author: Zhihua Liang
Github: https://github.com/magicknight
Date: 2021-07-20 09:18:17
LastEditors: Zhihua Liang
LastEditTime: 2021-07-20 09:18:18
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/losses/mtl.py
'''
import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat
from deepspace.graphs.losses.ssim import SSIM_Loss
from commontools.setup import config


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, emb_i, emb_j):

        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(repeat(representations, 'n d -> b n d', b=representations.shape[0]), repeat(representations, 'b d -> b n d', n=representations.shape[0]), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class LearnedLoss():
    def __init__(self, losstype, device, batch_size=None):

        if losstype == 'CrossEntropy':
            self.lossF = torch.nn.CrossEntropyLoss().to(device)
            self.adj = 1
        elif losstype == 'L1':
            # self.lossF = torch.nn.L1Loss().to(device)
            # self.adj = 0.5
            self.lossF = SSIM_Loss(data_range=1, channel=config.deepspace.image_channels, win_size=config.deepspace.ssim_win_size)
            self.adj = 0.5
        elif losstype == 'Contrastive':
            self.lossF = ContrastiveLoss(batch_size, device=device).to(device)
            self.adj = 1

    def calculate_loss(self, output, label):
        return self.lossF(output, label)

    def calculate_weighted_loss(self, loss, _s):
        return self.adj * torch.exp(-_s) * loss + 0.5 * _s


class MTLLOSS():
    def __init__(self, loss_funcs, device):
        super().__init__()
        self._loss_funcs = loss_funcs
        self.device = device

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive,
                 output_recons, target_recons, rot_w, contrastive_w, reconstruction_w):
        """Returns (overall loss, [seperate task losses])"""

        r_loss = self._loss_funcs[0].calculate_loss(output_rot, target_rot)
        rotation_loss = self._loss_funcs[0].calculate_weighted_loss(r_loss, rot_w)

        cn_loss = self._loss_funcs[1].calculate_loss(output_contrastive, target_contrastive)
        contrastive_loss = self._loss_funcs[1].calculate_weighted_loss(cn_loss, contrastive_w)

        rec_loss = self._loss_funcs[2].calculate_loss(output_recons, target_recons)
        reconstruction_loss = self._loss_funcs[2].calculate_weighted_loss(rec_loss, reconstruction_w)

        total_loss = rotation_loss + contrastive_loss + reconstruction_loss
        total_loss = total_loss.squeeze()

        return total_loss, (r_loss, cn_loss, rec_loss)


def MTL_loss(device, batch_size):
    """Returns the learned uncertainty loss function."""

    task_rot = LearnedLoss('CrossEntropy', device, batch_size)
    task_contrastive = LearnedLoss('Contrastive', device, batch_size)
    task_recons = LearnedLoss('L1', device, batch_size)
    return MTLLOSS([task_rot, task_contrastive, task_recons], device)
