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
Date: 2021-08-12 20:01:59
LastEditors: Zhihua Liang
LastEditTime: 2021-08-13 08:31:51
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/models/transformer/vit_ae.py
'''


import torch
from torch import nn
from einops.layers.torch import Rearrange
import timm
from timm.models.layers import PatchEmbed, Mlp


class VITAE(nn.Module):
    def __init__(self, image_size, patch_size, model_name='vit_base_patch16_384', in_channels=3, out_channel=1, pretrained=False, embed_dim=768, drop_rate=0.0,):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        n_patches = image_size // patch_size
        # self.num_tokens = 1
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.vit = timm.create_model(model_name, pretrained=pretrained, in_chans=in_channels)
        if pretrained:
            for p in self.vit.parameters():
                p.requires_grad = False
        self.norm = nn.LayerNorm(embed_dim)
        self.rearrange = Rearrange('b (h w) c -> b c h w', h=n_patches)
        self.to_img = nn.ConvTranspose2d(embed_dim, out_channel, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patch_embed(x)
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_token, x), dim=1)
        # x = self.pos_drop(x + self.pos_embed)
        x = self.vit.blocks(x)
        x = self.norm(x)
        x = self.rearrange(x)
        x = self.to_img(x)


if __name__ == "__main__":
    from torchinfo import summary
    model = VITAE(image_size=768, patch_size=32, pretrained=True)
    summary(model, input_size=(6, 3, 768, 768))
