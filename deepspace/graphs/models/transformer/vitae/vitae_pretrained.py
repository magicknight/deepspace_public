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
Date: 2021-08-12 20:01:59
LastEditors: Zhihua Liang
LastEditTime: 2021-08-14 21:38:05
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/models/transformer/vitae/vitae.py
'''


import torch
from torch import nn
from deepspace.graphs.models.transformer.vitae.encoder import Encoder
from deepspace.graphs.models.transformer.vitae.decoder import Decoder


class VITAE(nn.Module):
    def __init__(self, image_size, patch_size, encoder_name='vit_base_patch16_384', in_channels=3, out_channel=1, pretrained=False, embed_dim=768, drop_rate=0.0,):
        super().__init__()
        self.encoder = Encoder(image_size, patch_size, encoder_name, in_channels, pretrained, embed_dim, drop_rate)
        self.to_img = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, out_channel, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.to_img(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    model = VITAE(image_size=768, patch_size=32, pretrained=True)
    summary(model, input_size=(6, 3, 768, 768))
    test_data = torch.randn(6, 3, 768, 768)
    out = model(test_data)
    print(out.shape, out.min(), out.max(), out.mean())
