#!/usr/bin/env bash

#use this line to run the main.py file with a specified config file
#python3 main.py PATH_OF_THE_CONFIG_FILE

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=1
#python main.py configs/dqn_exp_0.json
#python main.py configs/dcgan_exp_0.json
#python main.py configs/condensenet_exp_0.json
#python main.py configs/mnist_exp_0.json
# python3 main.py configs/erfnet_exp_0.json
# python3 main.py configs/toml/defect/defect_detection_train_defect_images.toml
# python3 main.py configs/toml/defect/defect_detection_train_normal_images.toml

# restoration
python3 main.py configs/toml/defect/restoration.toml

# wp8 
# python3 main.py configs/toml/defect/wp8_autoencoder_same_data.toml



