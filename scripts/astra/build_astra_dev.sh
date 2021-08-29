#!/bin/bash
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
 # @Description: 
 # @Author: Zhihua Liang
 # @Github: https://github.com/magicknight
 # @Date: 2021-08-26 17:15:59
 # @LastEditors: Zhihua Liang
 # @LastEditTime: 2021-08-26 17:16:00
 # @FilePath: /home/zhihua/framework/deepspace/scripts/astra/build_astra_dev.sh
###

cmake -DBOOST_ROOT=/usr/include/boost \
      -DOptiX_INSTALL_DIR=/home/zhihua/soft/NVIDIA-OptiX-SDK-6.5.0-linux64 \
      -DCMAKE_CXX_FLAGS="-DASTRA_PYTHON -I/usr/include/python3.8 -I/home/zhihua/soft/NVIDIA-OptiX-SDK-6.5.0-linux64/include" \
      -DWITH_PYTHON=ON \
      -DPYTHON_LIBRARY=/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.so \
      -DPYTHON_EXECUTABLE=/usr/bin/python3.8 \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DWITH_MATLAB=OFF \
      .

