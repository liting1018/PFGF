# Pseudo Visible Feature Fine-Grained Fusion for Thermal Object Detection (CVPR-25)
**Note:** ![](./figs/almost.png) This repository currently provides **inference code**. Full training code is almost there! ![](./figs/almost.png)
## Overview
![](./figs/overview.png)
## Environment Configuration
1. Create Virtual Environment
```
conda create -n PFGF_Inference python=3.9
conda activate PFGF_Inference
```
2. Install PyTorch
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
3. Install MMCV using MIM
```
pip install -U openmim
mim install mmcv-full==1.7.2
```
4. Install MMDetection
```
git clone https://github.com/liting1018/PFGF.git
cd PFGF_Inference
pip install -r requirements/build.txt
pip install -v -e .
```
## Quick Inference
For FLIR dataset, please run
```
python tools/analysis_tools/eval_metric.py configs/yolox_l_tirgraphmamba_1x8_200e_FLIR_r_test.py ./pkl/FLIR.pkl --eval mAP
```

For LLVIP dataset, please run
```
python tools/analysis_tools/eval_metric.py configs/yolox_l_tirgraphmamba_1x8_200e_LLVIP_r_test.py ./pkl/LLVIP.pkl --eval mAP
``` 

For Autonomous Vehicles dataset, please run
```
python tools/analysis_tools/eval_metric.py configs/yolox_l_tirgraphmamba_1x8_200e_AV_r_test.py ./pkl/AV.pkl --eval mAP
``` 
