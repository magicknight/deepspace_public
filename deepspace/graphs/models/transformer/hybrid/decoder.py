'''
                       ::
                      :;J7, :,                        ::;7:
                      ,ivYi, ,                       ;LLLFS:
                      :iv7Yi                       :7ri;j5PL
                     ,:ivYLvr                    ,ivrrirrY2X,
                     :;r@Wwz.7r:                :ivu@kexianli.
                    :iL7::,:::iiirii:ii;::::,,irvF7rvvLujL7ur
                   ri::,:,::i:iiiiiii:i:irrv177JX7rYXqZEkvv17
                ;i:, , ::::iirrririi:i:::iiir2XXvii;L8OGJr71i
              :,, ,,:   ,::ir@mingyi.irii:i:::j1jri7ZBOS7ivv,
                 ,::,    ::rv77iiiriii:iii:i::,rvLq@huhao.Li
             ,,      ,, ,:ir7ir::,:::i;ir:::i:i::rSGGYri712:
           :::  ,v7r:: ::rrv77:, ,, ,:i7rrii:::::, ir7ri7Lri
          ,     2OBBOi,iiir;r::        ,irriiii::,, ,iv7Luur:
        ,,     i78MBBi,:,:::,:,  :7FSL: ,iriii:::i::,,:rLqXv::
        :      iuMMP: :,:::,:ii;2GY7OBB0viiii:i:iii:i:::iJqL;::
       ,     ::::i   ,,,,, ::LuBBu BBBBBErii:i:i:i:i:i:i:r77ii
      ,       :       , ,,:::rruBZ1MBBqi, :,,,:::,::::::iiriri:
     ,               ,,,,::::i:  @arqiao.       ,:,, ,:::ii;i7:
    :,       rjujLYLi   ,,:::::,:::::::::,,   ,:i,:,,,,,::i:iii
    ::      BBBBBBBBB0,    ,,::: , ,:::::: ,      ,,,, ,,:::::::
    i,  ,  ,8BMMBBBBBBi     ,,:,,     ,,, , ,   , , , :,::ii::i::
    :      iZMOMOMBBM2::::::::::,,,,     ,,,,,,:,,,::::i:irr:i:::,
    i   ,,:;u0MBMOG1L:::i::::::  ,,,::,   ,,, ::::::i:i:iirii:i:i:
    :    ,iuUuuXUkFu7i:iii:i:::, :,:,: ::::::::i:i:::::iirr7iiri::
    :     :rk@Yizero.i:::::, ,:ii:::::::i:::::i::,::::iirrriiiri::,
     :      5BMBBBBBBSr:,::rv2kuii:::iii::,:i:,, , ,,:,:i@petermu.,
          , :r50EZ8MBBBBGOBBBZP7::::i::,:::::,: :,:,::i;rrririiii::
              :jujYY7LS0ujJL7r::,::i::,::::::::::::::iirirrrrrrr:ii:
           ,:  :@kevensun.:,:,,,::::i:i:::::,,::::::iir;ii;7v77;ii;i,
           ,,,     ,,:,::::::i:iiiii:i::::,, ::::iiiir@xingjief.r;7:i,
        , , ,,,:,,::::::::iiiiiiiiii:,:,:::::::::iiir;ri7vL77rrirri::
         :,, , ::::::::i:::i:::i:i::,,,,,:,::i:i:::iir;@Secbone.ii:::

Description: 
Author: Zhihua Liang
Github: https://github.com/magicknight
Date: 2021-08-23 21:52:43
LastEditors: Zhihua Liang
LastEditTime: 2021-08-25 00:19:36
FilePath: /home/zhihua/framework/deepspace/deepspace/graphs/models/transformer/hybrid/decoder.py
'''


import torch
from torch import nn
from einops.layers.torch import Rearrange
from deepspace.graphs.layers.transformer.layers import Attention, MAttention, FeedForward, Residual, MResidual, nnResidual


class Decoder(nn.Module):
    def __init__(self, num_patches, channels, patch_size, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        dim = channels * patch_size * patch_size
        n_fts = num_patches * channels
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.ModuleList([
                    nnResidual(nn.MultiheadAttention(embed_dim=dim_head*heads, num_heads=heads, dropout=dropout)),
                    nn.LayerNorm(dim),
                ]),
                nn.ModuleList([
                    nnResidual(nn.MultiheadAttention(embed_dim=dim_head*heads, num_heads=heads, dropout=dropout)),
                    nn.LayerNorm(dim),
                ]),
                nn.Sequential(
                    Residual(FeedForward(dim, mlp_dim, dropout=dropout)),
                    nn.LayerNorm(dim),
                ),
                Rearrange('b n (c h w) -> b (n c) h w', c=channels, h=patch_size),
                nn.ConvTranspose2d(n_fts, n_fts, 4, stride=2, padding=1),
                Rearrange('b (n c) h w -> b n (c h w)', n=num_patches),
            ]))
            dim_head = dim_head * 4
            dim = dim * 4
            patch_size = patch_size * 2

    def forward(self, y, m):
        for attn, mattn, ff, r1, conv, r2, in self.layers:
            at, ln = attn
            y = at(y, y, y, need_weights=False)
            y = ln(y)
            att, lyn = mattn
            y = att(y, m, m, need_weights=False)
            y = lyn(y)
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
    model = Decoder(num_patches=576, channels=1, patch_size=4, depth=3, heads=16, dim_head=1, mlp_dim=3072, dropout=0.1)
    summary(model, input_size=[(6, 576, 16), (6, 576, 16)])
    memory_data = torch.randn(6, 576, 16)
    test_data = torch.randn(6, 576, 16)
    out = model(test_data, memory_data)
    print(out.shape, out.min(), out.max(), out.mean())
