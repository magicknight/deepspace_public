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
Date: 2021-08-30 07:30:27
LastEditors: Zhihua Liang
LastEditTime: 2021-08-30 07:30:27
FilePath: /dxray/home/zhihua/framework/deepspace/deepspace/graphs/models/transformer/dimension_transformer/layers.py
'''
import einops
import torch
import torch.nn as nn


class FastAttention(nn.Module):
    """
    https://github.com/wilile26811249/Fastformer-PyTorch/blob/main/Fastformer.py
    """

    def __init__(self, dim=3, decode_dim=16):
        super(FastAttention, self).__init__()
        # Generate weight for Wquery、Wkey and Wvalue
        self.to_qkv = nn.Linear(dim, decode_dim * 3, bias=False)
        self.weight_q = nn.Linear(dim, decode_dim, bias=False)
        self.weight_k = nn.Linear(dim, decode_dim, bias=False)
        self.weight_v = nn.Linear(dim, decode_dim, bias=False)
        self.weight_r = nn.Linear(decode_dim, decode_dim, bias=False)
        self.weight_alpha = nn.Parameter(torch.randn(decode_dim))
        self.weight_beta = nn.Parameter(torch.randn(decode_dim))
        self.scale_factor = decode_dim ** -0.5

    def forward(self, x, mask=None):
        query = self.weight_q(x)
        key = self.weight_k(x)
        value = self.weight_v(x)
        b, n, d = query.shape

        # Caculate the global query
        alpha_weight = torch.softmax(torch.mul(query, self.weight_alpha) * self.scale_factor, dim=-1)
        global_query = query * alpha_weight
        global_query = torch.einsum('b n d -> b d', global_query)

        # Model the interaction between global query vector and the key vector
        repeat_global_query = einops.repeat(global_query, 'b d -> b copy d', copy=n)
        p = repeat_global_query * key
        beta_weight = torch.softmax(torch.mul(p, self.weight_beta) * self.scale_factor, dim=-1)
        global_key = p * beta_weight
        global_key = torch.einsum('b n d -> b d', global_key)

        # key-value
        key_value_interaction = torch.einsum('b j, b n j -> b n j', global_key, value)
        key_value_interaction_out = self.weight_r(key_value_interaction)
        result = key_value_interaction_out + query
        return result


if __name__ == '__main__':
    model = FastAttention(dim=3, decode_dim=8)
    x = torch.randn(4, 6, 3)
    result = model(x)
    print(result.size())
