# CoactSeg: Learning from Heterogeneous Data for New Multiple Sclerosis Lesion Segmentation
by Yicheng Wu*, Zhonghua Wu, Hengcan Shi, Bjoern Picker, Winston Chong, and Jianfei Cai.

### News
```
<26.07.2023> Due to IP restrictions, the data sharing is suspended now.
<11.07.2023> We release the codes.
```

### Introduction
This repository is for our MICCAI 2023 paper: '[CoactSeg: Learning from Heterogeneous Data for New Multiple Sclerosis Lesion Segmentation](https://arxiv.org/pdf/2307.04513.pdf)' (Early Acceptance, top 14%).

### Environment
This repository is based on PyTorch 1.8.0, CUDA 11.1, and Python 3.8.10. All experiments in our paper were conducted on a single NVIDIA Tesla V100 GPU with an identical experimental setting. 

### Data Preparation
Please obtain the original public [MSSEG-2](https://portal.fli-iam.irisa.fr/msseg-2/data/) Dataset. Then, the [HD-BET](https://github.com/MIC-DKFZ/HD-BET) tool is used to extract the brain regions. We further apply the re-sampling and z-score normalization operations [here](https://github.com/ycwu1997/CoactSeg/blob/main/data/MSSEG2/h5/pre_processing.py). The data split is fixed and given in 'CoactSeg/data'.

### Usage
1. Clone the repository;
```
git clone https://github.com/ycwu1997/CoactSeg.git
```
2. Train the model;
```
sh train_mixed.sh
```
3. Test the model;
```
sh test_mixed.sh
```

### Citation
If our model is useful for your research, please consider citing:
```
@inproceedings{wu2023coact,
  title={CoactSeg: Learning from Heterogeneous Data for New Multiple Sclerosis Lesion Segmentation},
  author={Wu, Yicheng and Wu, Zhonghua and Shi, Hengcan and Picker, Bjoern and Chong, Winston and Cai, Jianfei},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2023},
  organization={Springer, Cham}
  }
```

### Issues
The current training stage is slow and there is a trick when generating the second-time-point all-lesion result on the MSSEG-2 dataset (see [lines](https://github.com/ycwu1997/CoactSeg/blob/main/code/utils/test_patch.py) 65-66). That's because two-time-point all-lesion labels are not available for the model training and the model cannot identify such slight all-lesion differences at different time points.

We are addressing the training efficiency and the input disentanglement problems. The improved CoactSeg model and original samples on our MS-23v1 dataset will be released in the future.

If any other questions, feel free to contact me at 'ycwueli@gmail.com'

### Acknowledgement:
This repository is based on our previous [MC-Net](https://github.com/ycwu1997/MC-Net). We here also appreciate the public repositories of [SNAC](https://github.com/marianocabezas/msseg2) and [Neuropoly](https://github.com/ivadomed/ms-challenge-2021), and also thanks for the efforts to collect and share the [MSSEG-2](https://portal.fli-iam.irisa.fr/msseg-2/) dataset and our MS-23v1 dataset from Alfred Health, Australia.
