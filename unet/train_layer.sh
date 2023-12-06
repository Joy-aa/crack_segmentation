#!/bin/bash

python train.py --data_dir /nfs/wj/DamCrack --model_dir checkpoints/layer5/nodeconv --model_type vgg16
python train.py --data_dir /nfs/wj/DamCrack --model_dir checkpoints/layer5/deconv --model_type vgg16 --deconv
python train.py --data_dir /nfs/wj/DamCrack --model_dir checkpoints/layer4/nodeconv --model_type vgg16V2
python train.py --data_dir /nfs/wj/DamCrack --model_dir checkpoints/layer4/deconv --model_type vgg16V2 --deconv
python train.py --data_dir /nfs/wj/DamCrack --model_dir checkpoints/layer3/nodeconv --model_type vgg16V3
python train.py --data_dir /nfs/wj/DamCrack --model_dir checkpoints/layer3/deconv --model_type vgg16V3 --deconv