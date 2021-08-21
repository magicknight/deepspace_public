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
from einops import repeat
from einops.layers.torch import Rearrange
from torch.nn.modules.container import Sequential
from deepspace.graphs.layers.transformer.layers import Attention, FeedForward, Residual


class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Sequential(
                    Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    nn.LayerNorm(dim),
                ),
                nn.Sequential(
                    Residual(FeedForward(dim, mlp_dim, dropout=dropout)),
                    nn.LayerNorm(dim),
                ),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    model = Encoder(dim=768, depth=12, heads=12, dim_head=64, mlp_dim=3072, dropout=0.1)
    summary(model, input_size=(6, 577, 768))
    test_data = torch.randn(6, 577, 768)
    out = model(test_data)
    print(out.shape, out.min(), out.max(), out.mean())
