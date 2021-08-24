#!/bin/sh
###
 #                        ::
 #                       :;J7, :,                        ::;7:
 #                       ,ivYi, ,                       ;LLLFS:
 #                       :iv7Yi                       :7ri;j5PL
 #                      ,:ivYLvr                    ,ivrrirrY2X,
 #                      :;r@Wwz.7r:                :ivu@kexianli.
 #                     :iL7::,:::iiirii:ii;::::,,irvF7rvvLujL7ur
 #                    ri::,:,::i:iiiiiii:i:irrv177JX7rYXqZEkvv17
 #                 ;i:, , ::::iirrririi:i:::iiir2XXvii;L8OGJr71i
 #               :,, ,,:   ,::ir@mingyi.irii:i:::j1jri7ZBOS7ivv,
 #                  ,::,    ::rv77iiiriii:iii:i::,rvLq@huhao.Li
 #              ,,      ,, ,:ir7ir::,:::i;ir:::i:i::rSGGYri712:
 #            :::  ,v7r:: ::rrv77:, ,, ,:i7rrii:::::, ir7ri7Lri
 #           ,     2OBBOi,iiir;r::        ,irriiii::,, ,iv7Luur:
 #         ,,     i78MBBi,:,:::,:,  :7FSL: ,iriii:::i::,,:rLqXv::
 #         :      iuMMP: :,:::,:ii;2GY7OBB0viiii:i:iii:i:::iJqL;::
 #        ,     ::::i   ,,,,, ::LuBBu BBBBBErii:i:i:i:i:i:i:r77ii
 #       ,       :       , ,,:::rruBZ1MBBqi, :,,,:::,::::::iiriri:
 #      ,               ,,,,::::i:  @arqiao.       ,:,, ,:::ii;i7:
 #     :,       rjujLYLi   ,,:::::,:::::::::,,   ,:i,:,,,,,::i:iii
 #     ::      BBBBBBBBB0,    ,,::: , ,:::::: ,      ,,,, ,,:::::::
 #     i,  ,  ,8BMMBBBBBBi     ,,:,,     ,,, , ,   , , , :,::ii::i::
 #     :      iZMOMOMBBM2::::::::::,,,,     ,,,,,,:,,,::::i:irr:i:::,
 #     i   ,,:;u0MBMOG1L:::i::::::  ,,,::,   ,,, ::::::i:i:iirii:i:i:
 #     :    ,iuUuuXUkFu7i:iii:i:::, :,:,: ::::::::i:i:::::iirr7iiri::
 #     :     :rk@Yizero.i:::::, ,:ii:::::::i:::::i::,::::iirrriiiri::,
 #      :      5BMBBBBBBSr:,::rv2kuii:::iii::,:i:,, , ,,:,:i@petermu.,
 #           , :r50EZ8MBBBBGOBBBZP7::::i::,:::::,: :,:,::i;rrririiii::
 #               :jujYY7LS0ujJL7r::,::i::,::::::::::::::iirirrrrrrr:ii:
 #            ,:  :@kevensun.:,:,,,::::i:i:::::,,::::::iir;ii;7v77;ii;i,
 #            ,,,     ,,:,::::::i:iiiii:i::::,, ::::iiiir@xingjief.r;7:i,
 #         , , ,,,:,,::::::::iiiiiiiiii:,:,:::::::::iiir;ri7vL77rrirri::
 #          :,, , ::::::::i:::i:::i:i::,,,,,:,::i:i:::iir;@Secbone.ii:::
 # 
 # @Description: 
 # @Author: Zhihua Liang
 # @Github: https://github.com/magicknight
 # @Date: 2021-08-24 07:08:49
 # @LastEditors: Zhihua Liang
 # @LastEditTime: 2021-08-24 07:09:32
 # @FilePath: /home/zhihua/framework/deepspace/update_vm.sh
###


rsync -Pav --delete -e "ssh -i $HOME/.ssh/google_compute_engine" . zhihua@35.204.70.242:/home/zhihua/framework/deepspace
