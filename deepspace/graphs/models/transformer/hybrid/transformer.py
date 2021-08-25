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
Date: 2021-08-23 21:52:43
LastEditors: Zhihua Liang
LastEditTime: 2021-08-25 00:29:21
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/models/transformer/hybrid/transformer.py
'''


import torch
from torch import nn
from deepspace.graphs.models.transformer.hybrid.encoder import Encoder
from deepspace.graphs.models.transformer.hybrid.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, num_patches, channels, patch_size, depth, heads, dim_head,  mlp_dim, dropout=0.):
        super().__init__()
        self.encoder = Encoder(num_patches, channels, patch_size, depth, heads, dim_head,  mlp_dim, dropout=dropout)
        patch_size = patch_size // 2**depth
        self.decoder = Decoder(num_patches, channels, patch_size, depth, heads, dim_head,  mlp_dim, dropout=dropout)

    def forward(self, x):
        m, x = self.encoder(x)
        # y = torch.zeros_like(x, requires_grad=True)
        x = self.decoder(x, m)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    model = Transformer(num_patches=576, channels=1, patch_size=32, depth=3, heads=16, dim_head=64, mlp_dim=3072, dropout=0.1)
    summary(model, input_size=[(6, 576, 1024)], depth=6)
    test_data = torch.randn(6, 576, 1024)
    out = model(test_data)
    print(out.shape, out.min(), out.max(), out.mean())
