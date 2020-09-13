# Pixel-in-Pixel Net: Towards Efficient Facial Landmark Detection in the Wild
## Introduction
This is the code of paper [Pixel-in-Pixel Net: Towards Efficient Facial Landmark Detection in the Wild](https://arxiv.org/abs/2003.03771). We propose a novel facial landmark detector, PIPNet, that is **fast**, **accurate**, and **robust**. PIPNet can be trained under two settings: (1) supervised learning; (2) unsupervised domain adaptation (UDA). With UDA, PIPNet has better cross-domain generalization performance by utilizing massive amounts of unlabeled data. 

<img src="images/detection_heads.png" alt="det_heads" width="512px">

## Installation
1. Install PyTorch >= v1.1, <= v1.5
2. Clone this repository.
```Shell
git clone https://github.com/jhb86253817/PIPNet.git
```
3. Install the dependencies in requirements.txt
```Shell
pip install -r requirements.txt
```

## Demo
1. We use a [modified version](https://github.com/jhb86253817/FaceBoxesV2) of [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch) as the face detector, so go to folder `FaceBoxesV2/utils`, run `sh make.sh` to build for NMS.
2. For PIPNets, you can download our trained models from [here](https://drive.google.com/drive/folders/17OwDgJUfuc5_ymQ3QruD8pUnh5zHreP2?usp=sharing), and put them under folder `snapshots/DATA_NAME/EXPERIMENT_NAME/`. 
3. Back to folder `PIPNet`, edit `run_demo.sh` to choose the config file and input source you want, then run `sh run_demo.sh`. We support image, video, and camera as the input. Some sample predictions can be seen as follows.
* PIPNet-ResNet18 trained on WFLW, with image `images/1.jpg` as the input:
<img src="images/1_out_WFLW_model.jpg" alt="1_out_WFLW_model" width="400px">

* PIPNet-ResNet18 trained on WFLW, with video `videos/002.avi` as the input:
<img src="videos/002_out_WFLW_model.gif" alt="002_out_WFLW_model" width="512px">

* PIPNet-ResNet18 trained on 300W+CelebA (UDA), with video `videos/007.avi` as the input:
<img src="videos/007_out_300W_CELEBA_model.gif" alt="007_out_300W_CELEBA_model" width="512px">

## Training

### Supervised Learning
Datasets: [300W](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/), [COFW](http://www.vision.caltech.edu/xpburgos/ICCV13/), [WFLW](https://wywu.github.io/projects/LAB/WFLW.html), [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)

1. Download the datasets from official sources, then put them under folder `data`. The folder structure should look like this:
````
PIPNet
-- lib
-- experiments
-- data
   |-- data_300W
       |-- afw
       |-- helen
       |-- ibug
       |-- lfpw
   |-- COFW
       |-- COFW_train_color.mat
       |-- COFW_test_color.mat
   |-- WFLW
       |-- WFLW_images
       |-- WFLW_annotations
   |-- AFLW
       |-- flickr
       |-- AFLWinfo_release.mat
````
2. Go to folder `lib`, preprocess a dataset by running ```python preprocess.py DATA_NAME```. For example, to process 300W:
```
python preprocess.py data_300W
```
3. Back to folder `PIPNet`, edit `run_train.sh` to choose the config file you want. Then, train the model by running:
```
sh run_train.sh
```

### Unsupervised Domain Adaptation
Datasets: 
* data_300W_COFW_WFLW: 300W + COFW-68 (unlabeled) + WFLW-68 (unlabeled) 
* data_300W_CELEBA: 300W + CelebA (unlabeled)

1. Download 300W, COFW, and WFLW as in the supervised learning setting. Download annotations of COFW-68 test from [here](https://github.com/golnazghiasi/cofw68-benchmark). For 300W+CelebA, you also need to download the in-the-wild CelebA images from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), and the [face bounding boxes](https://drive.google.com/drive/folders/17OwDgJUfuc5_ymQ3QruD8pUnh5zHreP2?usp=sharing) detected by us. The folder structure should look like this:
````
PIPNet
-- lib
-- experiments
-- data
   |-- data_300W
       |-- afw
       |-- helen
       |-- ibug
       |-- lfpw
   |-- COFW
       |-- COFW_train_color.mat
       |-- COFW_test_color.mat
   |-- WFLW
       |-- WFLW_images
       |-- WFLW_annotations
   |-- data_300W_COFW_WFLW
       |-- cofw68_test_annotations
       |-- cofw68_test_bboxes.mat
   |-- CELEBA
       |-- img_celeba
       |-- celeba_bboxes.txt
   |-- data_300W_CELEBA
       |-- cofw68_test_annotations
       |-- cofw68_test_bboxes.mat
````
2. Go to folder `lib`, preprocess a dataset by running ```python preprocess_uda.py DATA_NAME```.
   To process data_300W_COFW_WFLW, run
   ```
   python preprocess_uda.py data_300W_COFW_WFLW
   ```
   To process data_300W_CELEBA, run
   ```
   python preprocess_uda.py CELEBA
   ```
   and
   ```
   python preprocess_uda.py data_300W_CELEBA
   ```
3. Back to folder `PIPNet`, edit `run_train.sh` to choose the config file you want. Then, train the model by running:
```
sh run_train.sh
```

## Evaluation
1. Edit `run_test.sh` to choose the config file you want. Then, test the model by running:
```
sh run_test.sh
```

## Acknowledgement
We thank the following great works:
* [human-pose-estimation.pytorch](https://github.com/microsoft/human-pose-estimation.pytorch)
* [HRNet-Facial-Landmark-Detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)
