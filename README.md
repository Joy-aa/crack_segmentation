# Crack Segmentation

<<<<<<< HEAD
=======
## RUN
tensorboard --logdir=/home/wj/local/crack_segmentation/runs --bind_all --port=6008

>>>>>>> 78eac1cd7246375db034e2b514d7439d4ace1cb1
## 1. Inference
- step 1. download the pre-trained model [unet_vgg16](https://drive.google.com/open?id=1wA2eAsyFZArG3Zc9OaKvnBuxSAPyDl08).
- step 2. put the downloaded model under the folder ./models
- step 3. run the code
```pythonstub
python inference.py  --img_dir /mnt/hangzhou_116_homes/DamDetection/data/Agray --model_path ../models/model_unet_vgg_16_best.pt --out_pred_dir ./result3
```

***
## 2. Training
- step 1. download the dataset from [the link](https://drive.google.com/open?id=1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP)
- step 2. run the training code
- step 3: run the code
```python 
python train_unet.py --data_dir your_data_path --model_dir your_model_path --model_type vgg_16
```

***
## 3. Reference

>https://github.com/alexdonchuk/cracks_segmentation_dataset

>https://github.com/yhlleo/DeepCrack

>https://github.com/ccny-ros-pkg/concreteIn_inpection_VGGF
