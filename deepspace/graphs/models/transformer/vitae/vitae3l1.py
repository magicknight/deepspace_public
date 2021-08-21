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
from einops.layers.torch import Rearrange
from deepspace.graphs.models.transformer.vitae import Transformer


class VITAE(nn.Module):
    def __init__(self, image_size, patch_size, dim=768, depth=12, heads=12, in_channels=3, out_channels=1, dim_head=64, dropout=0., emb_dropout=0., scale_dim=4):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        h = image_size // patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.to_img = nn.Sequential(
            nn.Linear(dim, patch_dim * out_channels // in_channels),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=h, p1=patch_size, p2=patch_size, c=out_channels),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)

        x += self.pos_embedding
        x = self.dropout(x)
        y = torch.zeros_like(x, requires_grad=True)
        x = self.transformer(x, y)

        return self.to_img(x)


if __name__ == "__main__":
    from torchinfo import summary
    # model = Transformer(dim=768, depth=12, heads=12, dim_head=64, mlp_dim=3072, dropout=0.1)
    # summary(model, input_size=[(6, 577, 768), (6, 577, 768)])

    # model = VITAE(image_size=768, patch_size=32, dim=768, depth=12, heads=12, in_channels=3, out_channels=1, dim_head=64, dropout=0., emb_dropout=0., scale_dim=4)
    model = VITAE(image_size=768, patch_size=32, dim=768, depth=1, heads=18, in_channels=3, out_channels=1, dim_head=32, dropout=0., emb_dropout=0., scale_dim=2)
    summary(model, input_size=[(4, 3, 768, 768)])
    test_data = torch.randn(1, 3, 768, 768)
    out = model(test_data)
    print(out.shape, out.min(), out.max(), out.mean())
