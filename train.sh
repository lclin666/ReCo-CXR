#!/bin/sh
CUDA_VISIBLE_DEVICES=1,3 python main_moco.py -a resnet50 --lr 0.03 --batch-size 128 \
--dist-url tcp://localhost:10005 \
--multiprocessing-distributed \
--world-size 1 --rank 0 \
--mlp --moco-t 0.2 --aug-plus --cos \
--epochs 400 /share2/home/wbq/lcl/datasets/CheXpert/
