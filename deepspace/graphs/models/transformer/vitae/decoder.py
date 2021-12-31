"""
                       _oo0oo_
                      o8888888o
                      88" . "88
                      (| -_- |)
                      0\  =  /0
                    ___/`---'\___
                  .' \\|     |// '.
                 / \\|||  :  |||// \
                / _||||| -:- |||||- \
               |   | \\\  - /// |   |
               | \_|  ''\---/''  |_/ |
               \  .-\__  '-'  ___/-. /
             ___'. .'  /--.--\  `. .'___
          ."" '<  `.___\_<|>_/___.' >' "".
         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
         \  \ `_.   \_ __\ /__ _/   .-` /  /
     =====`-.____`.___ \_____/___.-`___.-'=====
                       `=---='


     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

           佛祖保佑       永不宕机     永无BUG

Description: 
Author: Zhihua Liang
Github: https://github.com/magicknight
Date: 2021-08-18 14:25:58
LastEditors: Zhihua Liang
LastEditTime: 2021-08-24 23:37:02
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/models/transformer/vitae/decoder.py
"""


import torch
from torch import nn
from torch.nn.modules.normalization import LayerNorm
from deepspace.graphs.layers.transformer.layers import Attention, MAttention, FeedForward, Residual, MResidual, nnResidual


class Decoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                # Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                                nnResidual(nn.MultiheadAttention(embed_dim=dim_head * heads, num_heads=heads, dropout=dropout)),
                                nn.LayerNorm(dim),
                            ]
                        ),
                        nn.ModuleList(
                            [
                                # MResidual(MAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                                nnResidual(nn.MultiheadAttention(embed_dim=dim_head * heads, num_heads=heads, dropout=dropout)),
                                nn.LayerNorm(dim),
                            ]
                        ),
                        nn.Sequential(
                            Residual(FeedForward(dim, mlp_dim, dropout=dropout)),
                            nn.LayerNorm(dim),
                        ),
                    ]
                )
            )

    def forward(self, y, m):
        for attn, mattn, ff in self.layers:
            at, ln = attn
            y = at(y, y, y, need_weights=False)
            # y = at(y)
            y = ln(y)
            att, lyn = mattn
            y = att(m, m, y, need_weights=False)
            # y = att(torch.cat([y, m, m], dim=-1))
            y = lyn(y)
            y = ff(y)
        return y


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    model = Decoder(dim=768, depth=12, heads=12, dim_head=64, mlp_dim=3072, dropout=0.1)
    summary(model, input_size=[(6, 577, 768), (6, 577, 768)])
    test_data = torch.randn(6, 577, 768)
    out = model(test_data, test_data)
    print(out.shape, out.min(), out.max(), out.mean())
