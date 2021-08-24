'''
                               |~~~~~~~|
                               |       |
                               |       |
                               |       |
                               |       |
                               |       |
    |~.\\\_\~~~~~~~~~~~~~~xx~~~         ~~~~~~~~~~~~~~~~~~~~~/_//;~|
    |  \  o \_         ,XXXXX),                         _..-~ o /  |
    |    ~~\  ~-.     XXXXX`)))),                 _.--~~   .-~~~   |
     ~~~~~~~`\   ~\~~~XXX' _/ ';))     |~~~~~~..-~     _.-~ ~~~~~~~
              `\   ~~--`_\~\, ;;;\)__.---.~~~      _.-~
                ~-.       `:;;/;; \          _..-~~
                   ~-._      `''        /-~-~
                       `\              /  /
                         |         ,   | |
                          |  '        /  |
                           \/;          |
                            ;;          |
                            `;   .       |
                            |~~~-----.....|
                           | \             \
                          | /\~~--...__    |
                          (|  `\       __-\|
                          ||    \_   /~    |
                          |)     \~-'      |
                           |      | \      '
                           |      |  \    :
                            \     |  |    |
                             |    )  (    )
                              \  /;  /\  |
                              |    |/   |
                              |    |   |
                               \  .'  ||
                               |  |  | |
                               (  | |  |
                               |   \ \ |
                               || o `.)|
                               |`\\) |
                               |       |
                               |       |

Description: 
Author: Zhihua Liang
Github: https://github.com/magicknight
Date: 2021-08-20 22:17:20
LastEditors: Zhihua Liang
LastEditTime: 2021-08-24 23:48:45
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
    model = Transformer(dim=512, depth=12, heads=16, dim_head=32, mlp_dim=3072, dropout=0.1)
    summary(model, input_size=[(6, 577, 512), (6, 577, 512)])
