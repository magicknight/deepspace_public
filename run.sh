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
 # @Description: script to run deepspace tasks
 # @Author: Zhihua Liang
 # @Github: https://github.com/magicknight
 # @Date: 2021-06-23 17:44:59
 # @LastEditors: Zhihua Liang
 # @LastEditTime: 2021-06-23 17:45:00
 # @FilePath: /dxray/home/zhihua/framework/deepspace/run.sh
###


#use this line to run the main.py file with a specified config file
#python3 main.py --config PATH_OF_THE_CONFIG_FILE

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=1
#python main.py configs/dqn_exp_0.json
#python main.py configs/dcgan_exp_0.json
#python main.py configs/condensenet_exp_0.json
#python main.py configs/mnist_exp_0.json
# python3 main.py --config configs/erfnet_exp_0.json
# python3 main.py --config configs/toml/defect/defect_detection_train_defect_images.toml
# python3 main.py --config configs/toml/defect/defect_detection_train_normal_images.toml

# restoration
# python3 main.py --config configs/toml/defect/restoration.toml

# gan
# python3 main.py --config configs/toml/defect/defect_gan.toml
# gan & projector
# python3 main.py --config configs/toml/defect/defect_gan_projector.toml

# wp8 
# python3 main.py --config configs/toml/defect/wp8_autoencoder_same_data.toml
# wp8 make plots
# python3 experiments/wp8/make_plots.py --config configs/toml/defect/wp8_autoencoder_same_data.toml
# wp8 gan break image
# python3 main.py --config configs/toml/wp8/gan.toml
# python3 main.py --config configs/toml/wp8/gan_test.toml
# python3 main.py --config configs/toml/wp8/autoencoder_npy.toml
# tpu
# PT_XLA_DEBUG=1 TF_CPP_LOG_THREAD_ID=1 TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=tensor=5,computation_client=5,xrt_computation_client=5,aten_xla_type=1 python main.py --config configs/toml/wp8/tpu.toml
# python main.py --config configs/toml/wp8/tpu.toml
# TF_CPP_MIN_LOG_LEVEL=4 XLA_USE_F16=1 python main.py --config configs/toml/wp8/tpu_gan3d_break.toml
# python main.py --config configs/toml/wp8/tpu_gan3d_break.toml > ~/temp/logs 2>&1 &
# TF_CPP_LOG_THREAD_ID=1 TF_CPP_MIN_LOG_LEVEL=3 python main.py --config configs/toml/wp8/tpu_gan3d_break.toml
# python main.py --config configs/toml/wp8/tpu_gan3d_break.toml
# python main.py --config configs/toml/wp8/tpu_gan3d_train_dis.toml
# python main.py --config configs/toml/wp8/tpu_gan3d_simple_break.toml
# python main.py --config configs/toml/wp8/tpu_gan3d_simple_break_test.toml
# python main.py --config configs/toml/wp8/tpu_gan3d_simple_break_data_aug.toml
# python main.py --config configs/toml/wp8/tpu_gan3d_simple_break_data_aug.toml > /home/zhihua/temp/tpu.logs 2>&1 &
# python main.py --config configs/toml/wp8/tpu_gan3d_simple_break_data_aug_test.toml
# python main.py --config configs/toml/wp8/tpu_gan3d_simple_break.toml > /home/zhihua/temp/tpu.logs 2>&1 &
# python main.py --config configs/toml/wp8/tpu_gan3d_train_dis.toml --log-cli-level=ERROR

# python main.py --config configs/toml/wp8/tpu_ae3d.toml
# python main.py --config configs/toml/wp8/tpu_ae3d_test.toml
# python main.py --config configs/toml/wp8/tpu_ae3d.toml > /home/zhihua/temp/tpu.logs 2>&1 &
# python main.py --config configs/toml/wp8/tpu_ae3d_cyl9.toml > /home/zhihua/temp/tpu.logs 2>&1 &
# python main.py --config configs/toml/wp8/tpu_ae3d_ring.toml
# python main.py --config configs/toml/wp8/tpu_ae3d_cyl9.toml
# python main.py --config configs/toml/wp8/tpu_ae3d_cyl9_test.toml
# python main.py --config configs/toml/wp8/tpu_ae3d_cyl8.toml
# python main.py --config configs/toml/wp8/tpu_ae3d_cyl8_test.toml

# wp8 data
# wp8 data
# python experiments/wp8/make_dataset_parallel.py --config configs/toml/wp8/wp8_data_npy.toml
# python experiments/wp8/make_dataset_to_npy.py --config configs/toml/wp8/wp8_data_npy.toml
# python experiments/wp8/make_dataset_parallel.py --config configs/toml/wp8/wp8_data_npy_test.toml
# python experiments/wp8/make_dataset_tif_parallel.py --config configs/toml/wp8/wp8_data_tif_npy.toml
# python experiments/wp8/make_dataset_parallel_ring.py --config configs/toml/wp8/wp8_data_npy_test.toml 

# dqn
# python3 main.py --config configs/toml/poc2020/control/dqn.toml
# python3 main.py --config configs/toml/defect/metrics_dqn.toml


# bone
# python3 experiments/bone/make_dataset_parallel.py --config configs/toml/bone/make_data_npy.toml
# 3d
# python3 main.py --config configs/toml/bone/gan3d.toml
# python3 main.py --config configs/toml/bone/gan3d_test.toml
# python main.py --config configs/toml/bone/tpu_sit3d.toml
# XLA_USE_F16=0 python main.py --config configs/toml/bone/tpu_sit3d_test.toml

# 2d
# python3 main.py --config configs/toml/bone/tpu_vitae_2d3l1.toml
# python3 main.py --config configs/toml/bone/tpu_vitae_2d3l1.toml > /home/zhihua/temp/bone.log 2>&1 &
# python3 main.py --config configs/toml/bone/tpu_hybrid.toml
# python3 main.py --config configs/toml/bone/tpu_hybrid.toml > /home/zhihua/temp/bone.log 2>&1 &
# python3 main.py --config configs/toml/bone/tpu_mixaer.toml
# python3 main.py --config configs/toml/bone/tpu_mixaer.toml  > /home/zhihua/temp/mixaer.log 2>&1 &
# python3 main.py --config configs/toml/bone/gpu_gan2d.toml
# python3 main.py --config configs/toml/bone/tpu_gan2d_aug.toml
# python3 main.py --config configs/toml/bone/tpu_gan2d_aug_test.toml
# python3 main.py --config configs/toml/bone/tpu_gan2d_unspv.toml
python3 main.py --config configs/toml/bone/tpu_gan2d_unspv_test.toml
# python3 main.py --config configs/toml/bone/tpu_gan2d_unspv.toml  > /home/zhihua/temp/tpu_gan2d_unspv.log 2>&1 &
# python3 main.py --config configs/toml/bone/tpu_gan2d_aug.toml > /home/zhihua/temp/bone.log 2>&1 &
# python3 main.py --config configs/toml/bone/tpu_vitae_2d.toml
# python3 main.py --config configs/toml/bone/tpu_vitae_2d.toml > /home/zhihua/temp/bone.log 2>&1 &
# python main.py --config configs/toml/bone/tpu_sit2d.toml
# python main.py --config configs/toml/bone/tpu_sit2d.toml > /home/zhihua/temp/train_sit2d.logs 2>&1 &
# python main.py --config configs/toml/bone/gan2d.toml
# python main.py --config configs/toml/bone/gan2d_test.toml
# python main.py --config configs/toml/bone/tpu_gan3d_break.toml
# python main.py --config configs/toml/bone/tpu_gan3d_break_test.toml
# python main.py --config configs/toml/bone/tpu_gan3d_break_aug.toml
# python main.py --config configs/toml/bone/tpu_gan3d_break_aug_test.toml
# python main.py --config configs/toml/bone/tpu_ae3d.toml
# python main.py --config configs/toml/bone/tpu_ae3d_test.toml
# bone data
# python experiments/bone/tif_to_npy.py --config configs/toml/bone/data_tif_npy.toml 


# tsinghua
# make data
# python3 experiments/competition/tsinghua/make_dataset_final.py --config configs/toml/tsinghua/make_dataset.toml
# python3 experiments/competition/tsinghua/make_dataset_problem.py --config configs/toml/tsinghua/make_dataset_problem.toml
# run
# python3 main.py --config configs/toml/tsinghua/train.toml
# python3 main.py --config configs/toml/tsinghua/train_position.toml
# python3 main.py --config configs/toml/tsinghua/train_transformer.toml
# python3 main.py --config configs/toml/tsinghua/train_resnet.toml
# redict
# python3 main.py --config configs/toml/tsinghua/predict.toml
# python3 main.py --config configs/toml/tsinghua/predict_resnet.toml
# test
# python3 main.py --config configs/toml/tsinghua/test.toml
# python3 main.py --config configs/toml/tsinghua/test_resnet.toml


# tpu
# python main.py --config configs/toml/tpu/fairseq.toml
# imagenet training
# python main.py --config configs/toml/tpu/imagenet.toml
