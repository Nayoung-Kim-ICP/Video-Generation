# Dynamic Motion Estimation and Evolution Video Prediction Network

This repository is for DMEE introduced in the follow paper.

Nayoung Kim and Je-Won Kang, "Dynamic Motion Estimation and Evolution Video Prediction Network"

Introduction
=============

<img width="516" alt="model" src="https://user-images.githubusercontent.com/71854817/94213645-580d9200-ff12-11ea-8394-a8402c44c117.png">

Overall architecture of the dynamic motion estimation and evolution (DMEE) network.

Experiments
=============

<img width="403" alt="result1" src="https://user-images.githubusercontent.com/71854817/94213783-ba669280-ff12-11ea-8959-ca48001ae1c9.png">

Visual comparisons using “PushUps” dataset in UCF-101.

<img width="442" alt="result2" src="https://user-images.githubusercontent.com/71854817/94213787-bc305600-ff12-11ea-8e36-82008ec60c0b.PNG">

Visual comparisons using “Vehicle” dataset in YouTube-8M.

Environment
=============
GPU: NVIDIA TITIAN Xp

Ubuntu 16.04

python 3.6

tensorflow 1.14.0

numpy 1.18.4


Test
------------
python3 test_ucf101.py --gpu [GPU_number]
