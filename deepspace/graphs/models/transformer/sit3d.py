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

Description: orginal code for the transformer model: https://github.com/rishikksh20/SiT-pytorch/blob/master/sit.py
Author: Zhihua Liang
Github: https://github.com/magicknight
Date: 2021-07-19 09:45:55
LastEditors: Zhihua Liang
LastEditTime: 2021-07-19 09:45:56
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/models/transformer/sit.py
'''
from os import wait
from einops.einops import rearrange
import torch
from torch import nn
import numpy as np
from einops import repeat
from einops.layers.torch import Rearrange
from deepspace.graphs.layers.transformer.layers import Attention, PreNorm, FeedForward


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SiT3D(nn.Module):
    def __init__(self, image_size, patch_size, rotation_node, contrastive_head, dim=768, depth=12, heads=12, in_channels=1, dim_head=64, dropout=0., emb_dropout=0., scale_dim=4):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 3
        # patch_dim is the dimension of the patch embedding
        patch_dim = in_channels * patch_size ** 3
        # h = height
        # w = width
        h = image_size // patch_size
        w = h

        # rearrange the input to make it compatible with the transformer, 64x64x64 to 8x8x8, total of 512 patches, then transform to *dim* features
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (l p3) -> b (h w l) (p1 p2 p3 c)', p1=patch_size, p2=patch_size, p3=patch_size),
            nn.Linear(patch_dim, dim, bias=False),
        )
        # position embedding, number of positions is equal to the number of patches, add 4 for rotation and contrastive embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 4, dim))
        # rotation embedding, 1x3 rotation vector
        self.rotation_token = nn.Parameter(torch.randn(1, 3, dim))
        # contrastive embedding, 1x1 contrastive
        self.contrastive_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)
        # transfer back to the image space, the transformer output is 8x8x8, total of **dim** features, then reshape to **image_size**
        self.to_img = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w l) (p1 p2 p3 c) -> b c (h p1) (w p2) (l p3)', h=h, w=w, p1=patch_size, p2=patch_size, p3=patch_size, c=in_channels)
        )
        # rotation head, first normalize the input to be in the range of [-1, 1]
        self.rot_head = nn.Sequential(
            # why do we need to normalize the output?
            nn.LayerNorm(dim),
            nn.Linear(dim, rotation_node)
        )

        self.contrastive_head = nn.Sequential(
            # why do we need to normalize the output?
            nn.LayerNorm(dim),
            nn.Linear(dim, contrastive_head)
        )
        # create learnable parameters for the MTL task
        # 这是损失函数的权重
        self.rot_w = nn.Parameter(torch.tensor([1.0]))
        self.contrastive_w = nn.Parameter(torch.tensor([1.0]))
        self.recons_w = nn.Parameter(torch.tensor([1.0]))

    def forward(self, img):
        # 把3维的图像变成多个patch
        x = self.to_patch_embedding(img)
        # b = batch， n = num_patches
        b, n, _ = x.shape
        # rotation embedding broadcast to all batch
        rotation_tokens = repeat(self.rotation_token, '() n d -> b n d', b=b)
        # contrastive embedding broadcast to all batch
        contrastive_tokens = repeat(self.contrastive_token, '() n d -> b n d', b=b)
        # patches concatenate to rotation and contrastive embedding, added to dim=1, so the number of patches for each batch is n + 4 (rotation = 3, and contrastive = 1)
        x = torch.cat((rotation_tokens, contrastive_tokens, x), dim=1)
        # add pos_embedding to all patches
        x += self.pos_embedding[:, :(n + 4)]
        x = self.dropout(x)
        # output from the transformer, shape is b, n + 4, dim
        x = self.transformer(x)
        # rotation output, shape is b, 3, rotation_node
        l_rotation = self.rot_head(x[:, 0:3])
        l_rotation = rearrange(l_rotation, 'b n d -> b d n', b=b)
        l_contrastive = self.contrastive_head(x[:, 3])
        x = self.to_img(x[:, 4:])

        return l_rotation, l_contrastive, x, self.rot_w, self.contrastive_w, self.recons_w,


if __name__ == "__main__":
    img = torch.ones([1, 1, 64, 64, 64])

    model = SiT3D(image_size=64, patch_size=8, rotation_node=4, contrastive_head=512, dim=768, depth=6, heads=12, in_channels=1,
                  dim_head=16, dropout=0.1, emb_dropout=0., scale_dim=4)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    l_rotation, l_contrastive, out_img, w_rot, w_contrastive, w_recons = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
    print("Shape of l_rotation :", l_rotation.shape)  # [B, rotation_node]
    print("Shape of l_contrastive :", l_contrastive.shape)  # [B, contrastive_head]
    print("w_rot :", w_rot)  # [1, 1]
    print("w_contrastive :", w_contrastive)  # [1, 1]
    print("w_recons :", w_recons)  # [1, 1]
