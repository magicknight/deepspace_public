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


from torch import nn
from deepspace.graphs.models.transformer.vitae.encoder import Encoder
from deepspace.graphs.models.transformer.vitae.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.encoder = Encoder(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.decoder = Decoder(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)

    def forward(self, x, y):
        x = self.encoder(x)
        y = self.decoder(y, x)
        return y


if __name__ == "__main__":
    from torchinfo import summary
    model = Transformer(dim=768, depth=12, heads=12, dim_head=64, mlp_dim=3072, dropout=0.1)
    summary(model, input_size=[(6, 577, 768), (6, 577, 768)])
