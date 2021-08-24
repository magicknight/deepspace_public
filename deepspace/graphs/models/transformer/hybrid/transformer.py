'''
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
Date: 2021-08-20 22:17:20
LastEditors: Zhihua Liang
LastEditTime: 2021-08-20 22:44:50
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/models/transformer/vitae/transformer.py
'''

import torch
from torch import nn
from deepspace.graphs.models.transformer.hybrid.encoder import Encoder
from deepspace.graphs.models.transformer.hybrid.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, num_patches, channels, patch_size, depth, heads, dim_head,  mlp_dim, dropout=0.):
        super().__init__()
        self.encoder = Encoder(num_patches, channels, patch_size, depth, heads, dim_head,  mlp_dim, dropout=dropout)
        patch_size = patch_size // 2**depth
        self.decoder = Decoder(num_patches, channels, patch_size, depth, heads, dim_head,  mlp_dim, dropout=dropout)

    def forward(self, x):
        m, x = self.encoder(x)
        # y = torch.zeros_like(x, requires_grad=True)
        x = self.decoder(x, m)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    model = Transformer(num_patches=576, channels=1, patch_size=32, depth=3, heads=32, dim_head=64, mlp_dim=3072, dropout=0.1)
    summary(model, input_size=[(6, 576, 1024)], depth=6)
    test_data = torch.randn(6, 576, 1024)
    out = model(test_data)
    print(out.shape, out.min(), out.max(), out.mean())
