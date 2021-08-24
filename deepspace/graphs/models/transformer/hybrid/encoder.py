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
from deepspace.graphs.layers.transformer.layers import Attention, FeedForward, Residual


class Encoder(nn.Module):
    def __init__(self, num_patches, channels, patch_size, depth, heads, dim_head,  mlp_dim, dropout=0.):
        super().__init__()
        dim = channels * patch_size * patch_size
        n_fts = num_patches * channels
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            dim = dim // 4
            self.layers.append(nn.ModuleList([
                Rearrange('b n (c h w) -> b (n c) h w', c=channels, h=patch_size),
                nn.Conv2d(n_fts, n_fts, 4, stride=2, padding=1),
                Rearrange('b (n c) h w -> b n (c h w)', n=num_patches),
                nn.Sequential(
                    Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    nn.LayerNorm(dim),
                ),
                nn.Sequential(
                    Residual(FeedForward(dim, mlp_dim, dropout=dropout)),
                    nn.LayerNorm(dim),
                ),
            ]))
            patch_size = patch_size // 2

    def forward(self, x):
        m = x
        for r1, conv, r2, attn, ff, in self.layers:
            data = torch.cat([x, m], dim=0)
            data = r1(data)
            data = conv(data)
            data = r2(data)
            x, m = data.chunk(2, dim=0)
            x = attn(x)
            x = ff(x)
        return x, m


if __name__ == "__main__":
    from torchinfo import summary
    model = Encoder(num_patches=576, channels=1, patch_size=32, depth=3, heads=32, dim_head=64, mlp_dim=3072, dropout=0.1)
    summary(model, input_size=(6, 576, 1024))
    test_data = torch.randn(6, 576, 1024)
    out = model(test_data)
    print(out.shape, out.min(), out.max(), out.mean())
