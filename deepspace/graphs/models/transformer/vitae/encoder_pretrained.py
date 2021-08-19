'''
          佛曰:  
                  写字楼里写字间，写字间里程序员；  
                  程序人员写程序，又拿程序换酒钱。  
                  酒醒只在网上坐，酒醉还来网下眠；  
                  酒醉酒醒日复日，网上网下年复年。  
                  但愿老死电脑间，不愿鞠躬老板前；  
                  奔驰宝马贵者趣，公交自行程序员。  
                  别人笑我忒疯癫，我笑自己命太贱；  
                  不见满街漂亮妹，哪个归得程序员？

Description: 
Author: Zhihua Liang
Github: https://github.com/magicknight
Date: 2021-08-18 14:19:52
LastEditors: Zhihua Liang
LastEditTime: 2021-08-18 14:19:54
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/models/transformer/vitae/encoder.py
'''
import torch
from torch import nn
from einops.layers.torch import Rearrange
import timm
from timm.models.layers import PatchEmbed, Mlp
from torch.nn.modules.activation import Sigmoid


class Encoder(nn.Module):
    def __init__(self, image_size, patch_size, encoder_name='vit_base_patch16_384', in_channels=3, pretrained=True, embed_dim=768, drop_rate=0.0,):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        n_patches = image_size // patch_size
        self.num_tokens = 1
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, in_chans=in_channels)
        if pretrained:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.norm = nn.LayerNorm(embed_dim)
        self.rearrange = Rearrange('b (h w) c -> b c h w', h=n_patches)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder.blocks(x)
        x = self.norm(x)
        x = x[:, 1:]
        x = self.rearrange(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    model = Encoder(image_size=768, patch_size=32, pretrained=True)
    summary(model, input_size=(6, 3, 768, 768))
    test_data = torch.randn(6, 3, 768, 768)
    out = model(test_data)
    print(out.shape, out.min(), out.max(), out.mean())
