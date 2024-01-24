#!/bin/bash
python train_gate.py --data_dir /mnt/hangzhou_116_homes/DamDetection/data/cutDataset/overlap0.6_ts1000_slice224/total/ --model_dir checkpoints/crackseg9k/loss3 --r1 1 --r2 20  --lr 0.0005
python train_gate.py --data_dir /mnt/hangzhou_116_homes/DamDetection/data/cutDataset/overlap0.6_ts1000_slice224/total/ --model_dir checkpoints/crackseg9k/loss4 --r1 5 --r2 20 --lr 0.0005
python train_gate.py --data_dir /mnt/hangzhou_116_homes/DamDetection/data/cutDataset/overlap0.6_ts1000_slice224/total/ --model_dir checkpoints/crackseg9k/loss6 --r1 10 --r2 20  --lr 0.0005
python train_gate.py --data_dir /mnt/hangzhou_116_homes/DamDetection/data/cutDataset/overlap0.6_ts1000_slice224/total/ --model_dir checkpoints/crackseg9k/loss11 --r1 7 --r2 20 --lr 0.0005
python train_gate.py --data_dir /mnt/hangzhou_116_homes/DamDetection/data/cutDataset/overlap0.6_ts1000_slice224/total/ --model_dir checkpoints/crackseg9k/loss10 --r1 3 --r2 20 --lr 0.0005


# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss1 --r0 --r1 1 --r2 20
# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss2 --joint_edgeseg_loss
# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss3 --r1 1 --r2 20
# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss4 --r1 5 --r2 20
# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss5 --r1 1 --r2 10

# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss6 --r1 10 --r2 20 --lr 0.0005
# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss7 --r1 10 --r2 20 --normal --lr 0.0001

# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss8 --r1 5 --r2 10 --lr 0.0005
# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss9 --r1 5 --r2 10 --normal --lr 0.0001
# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss10 --r1 3 --r2 20
# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss11 --r1 7 --r2 20
# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss12 --r1 5 --r2 20 --snapshot checkpoints/crackseg9k/loss4/model_best.pt --normal --att_th 0.9 --lr 0.0005
# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss13 --r1 5 --r2 20 --snapshot checkpoints/crackseg9k/loss4/model_best.pt --normal --att_th 0.8 --lr 0.0005
# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss14 --r1 5 --r2 20 --snapshot checkpoints/crackseg9k/loss4/model_best.pt --normal --att_th 0.7 --lr 0.0005
# python train_gate.py --data_dir  /mnt/hangzhou_116_homes/zek/crackseg9k/ --model_dir checkpoints/crackseg9k/loss15 --r1 1 --r2 20 --snapshot checkpoints/crackseg9k/loss3/model_best.pt --normal --lr 0.0005

# python train.py --data_dir /nfs/wj/DamCrack --model_dir checkpoints/layer5/deconv --model_type vgg16 --deconv
# python train.py --data_dir /nfs/wj/DamCrack --model_dir checkpoints/layer4/nodeconv --model_type vgg16V2
# python train.py --data_dir /nfs/wj/DamCrack --model_dir checkpoints/layer4/deconv --model_type vgg16V2 --deconv
# python train.py --data_dir /nfs/wj/DamCrack --model_dir checkpoints/layer3/nodeconv --model_type vgg16V3
# python train.py --data_dir /nfs/wj/DamCrack --model_dir checkpoints/layer3/deconv --model_type vgg16V3 --deconv
