#!/usr/bin/env bash

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
python3 main.py --config configs/toml/wp8/gan.toml


# dqn
# python3 main.py --config configs/toml/reinforcement/dqn.toml
# python3 main.py --config configs/toml/defect/metrics_dqn.toml
