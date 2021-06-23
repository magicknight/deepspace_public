#!/usr/bin/env bash

python /home/zhihua/framework/xla/test/test_train_mp_imagenet.py --datadir=/home/zhihua/ssd/imagenet --model=resnet50 --num_epochs=90 --num_workers=8 --num_cores=8 --batch_size=128 --log_steps=200
