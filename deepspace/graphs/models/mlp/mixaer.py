'''
                  ___====-_  _-====___
            _--^^^#####//      \\#####^^^--_
         _-^##########// (    ) \\##########^-_
        -############//  |\^^/|  \\############-
      _/############//   (@::@)   \############\_
     /#############((     \\//     ))#############\
    -###############\\    (oo)    //###############-
   -#################\\  / VV \  //#################-
  -###################\\/      \//###################-
 _#/|##########/\######(   /\   )######/\##########|\#_
 |/ |#/\#/\#/\/  \#/\##\  |  |  /##/\#/  \/\#/\#/\#| \|
 `  |/  V  V  `   V  \#\| |  | |/#/  V   '  V  V  \|  '
    `   `  `      `   / | |  | | \   '      '  '   '
                     (  | |  | |  )
                    __\ | |  | | /__
                   (vvv(VVV)(VVV)vvv)

     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

               神兽保佑            永无BUG

Description: 
Author: Zhihua Liang
Github: https://github.com/magicknight
Date: 2021-08-21 22:55:29
LastEditors: Zhihua Liang
LastEditTime: 2021-08-21 22:55:55
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/models/mlp/mixaer.py
'''


import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class BottleNeck(nn.Module):
    def __init__(self, dim, hidden_dim, n, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            # Rearrange('b n c -> b (n c)'),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
            # Rearrange('b (n c) -> b n c', n=n),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

    def forward(self, x):
        x = x + self.channel_mix(x)
        x = x + self.token_mix(x)
        return x


class MLPMixaer(nn.Module):

    def __init__(self, in_channels, out_channels, dim, patch_size, image_size, depth, token_dim, channel_dim, bottleneck_dim, dropout=0.):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        h = image_size // patch_size
        self.num_patch = (image_size // patch_size) ** 2
        # self.to_patch_embedding = nn.Sequential(
        #     nn.Conv2d(in_channels, dim, patch_size, patch_size),
        #     Rearrange('b c h w -> b (h w) c'),
        # )
        in_patch_dim = in_channels * patch_size ** 2
        out_path_dim = out_channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(in_patch_dim, dim),
        )

        self.encoder_blocks = nn.ModuleList([])
        self.decoder_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.encoder_blocks.append(EncoderBlock(dim, self.num_patch, token_dim, channel_dim, dropout))

        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim), nn.LayerNorm(dim)])

        self.bottleneck = BottleNeck(dim, bottleneck_dim, self.num_patch, dropout)

        for _ in range(depth):
            self.decoder_blocks.append(DecoderBlock(dim, self.num_patch, token_dim, channel_dim, dropout))

        # self.to_image_embedding = nn.Sequential(
        #     Rearrange('b (h w) c -> b c h w', h=h),
        #     nn.ConvTranspose2d(dim, out_channels, patch_size, patch_size),
        #     # nn.Sigmoid()
        # )

        self.to_image_embedding = nn.Sequential(
            nn.Linear(dim, out_path_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size),
            nn.Sigmoid(),
        )

    def forward(self, x):

        x = self.to_patch_embedding(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        x = self.layer_norms[0](x)

        # x = self.bottleneck(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)

        x = self.layer_norms[1](x)

        x = self.to_image_embedding(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    img = torch.ones([1, 5, 768, 768])

    model = MLPMixaer(in_channels=5, out_channels=1, image_size=768, patch_size=32, dim=1024, depth=4, token_dim=256, channel_dim=512, bottleneck_dim=256, dropout=0.1)
    summary(model, input_size=[img.shape], depth=6)

    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    # print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
