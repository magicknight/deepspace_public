#!/usr/bin/env bash
###
 #  ┌─────────────────────────────────────────────────────────────┐
 #  │┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
 #  ││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
 #  │├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
 #  ││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
 #  │├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
 #  ││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
 #  │├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
 #  ││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
 #  │└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
 #  │      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
 #  │      └───┴─────┴───────────────────────┴─────┴───┘          │
 #  └─────────────────────────────────────────────────────────────┘
 # 
 # @Description: script to run video processing
 # @Author: Zhihua Liang
 # @Github: https://github.com/magicknight
 # @Date: 2021-06-23 17:44:59
 # @LastEditors: Zhihua Liang
 # @LastEditTime: 2021-06-23 17:45:00
 # @FilePath: /dxray/home/zhihua/framework/deepspace/run.sh
###

python3 scripts/video/animeGAN_TPU.py --config configs/toml/video/animeGAN_TPU.toml > /home/zhihua/temp/video.log 2>&1&

