'''

   ┏┓　　　┏┓
 ┏┛┻━━━┛┻┓
 ┃　　　　　　　┃
 ┃　　　━　　　┃
 ┃　＞　　　＜　┃
 ┃　　　　　　　┃
 ┃...　⌒　...　┃
 ┃　　　　　　　┃
 ┗━┓　　　┏━┛
     ┃　　　┃　
     ┃　　　┃
     ┃　　　┃
     ┃　　　┃  神兽保佑
     ┃　　　┃  代码无bug　　
     ┃　　　┃
     ┃　　　┗━━━┓
     ┃　　　　　　　┣┓
     ┃　　　　　　　┏┛
     ┗┓┓┏━┳┓┏┛
       ┃┫┫　┃┫┫
       ┗┻┛　┗┻┛

Description: 
Author: Zhihua Liang
Github: https://github.com/magicknight
Date: 2021-08-18 14:25:58
LastEditors: Zhihua Liang
LastEditTime: 2021-08-21 07:57:32
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/models/transformer/vitae/decoder.py
'''


import torch
from torch import nn
from einops.layers.torch import Rearrange
from deepspace.graphs.layers.transformer.layers import Attention, MAttention, FeedForward, Residual, MResidual


class Decoder(nn.Module):
    def __init__(self, num_patches, channels, patch_size, depth, heads, dim_head,  mlp_dim, dropout=0.):
        super().__init__()
        dim = channels * patch_size * patch_size
        n_fts = num_patches * channels
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Sequential(
                    Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    nn.LayerNorm(dim),
                ),
                nn.Sequential(
                    MResidual(MAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    nn.LayerNorm(dim),
                ),
                nn.Sequential(
                    Residual(FeedForward(dim, mlp_dim, dropout=dropout)),
                    nn.LayerNorm(dim),
                ),
                Rearrange('b n (c h w) -> b (n c) h w', c=channels, h=patch_size),
                nn.ConvTranspose2d(n_fts, n_fts, 4, stride=2, padding=1),
                Rearrange('b (n c) h w -> b n (c h w)', n=num_patches),
            ]))
            dim = dim * 4
            patch_size = patch_size * 2

    def forward(self, y, m):
        for attn, mattn, ff, r1, conv, r2, in self.layers:
            y = attn(y)
            y = mattn(torch.cat([y, m, m], dim=-1))
            y = ff(y)
            data = torch.cat([y, m], dim=0)
            data = r1(data)
            data = conv(data)
            data = r2(data)
            y, m = data.chunk(2, dim=0)
        return y


if __name__ == "__main__":
    import torch
    from torchinfo import summary
    model = Decoder(num_patches=576, channels=1, patch_size=4, depth=3, heads=32, dim_head=64, mlp_dim=3072, dropout=0.1)
    summary(model, input_size=[(6, 576, 16), (6, 576, 16)])
    memory_data = torch.randn(6, 576, 16)
    test_data = torch.randn(6, 576, 16)
    out = model(test_data, memory_data)
    print(out.shape, out.min(), out.max(), out.mean())
