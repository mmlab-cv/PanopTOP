# PanopTOP
Official code and dataset for the paper [**PanopTOP: a framework for generating viewpoint-invariant human pose
estimation datasets**](https://openaccess.thecvf.com/content/ICCV2021W/DSC/papers/Garau_PanopTOP_A_Framework_for_Generating_Viewpoint-Invariant_Human_Pose_Estimation_Datasets_ICCVW_2021_paper.pdf)

## Requirements
- numpy
- open3d
- pytorch
- matplotlib
- opencv-python

## Dataset download
The dataset can be found at [this Google Drive link](https://drive.google.com/file/d/1725yyYnqmcurE-iT2wH7eorVatWn9PYR/view?usp=sharing).

## Dataset structure
```
Dataset   
│
└───front
│   │
│   └───camera_params
│   │   │   *.json
│   │   │   *.json
│   │   │   ...
    │
│   └───train
│   │   │   *.npy
│   │   │   *.png
│   │   │   ...
    │
│   └───train2d
│   │   │   *.npy
│   │   │   *.npy
│   │   │   ...
    │
│   └───traindepth
│   │   │   *.png
│   │   │   *.png
│   │   │   ...
    │
│   └───validation
│   │   │   *.npy
│   │   │   *.png
│   │   │   ...
    │
│   └───validation2d
│   │   │   *.npy
│   │   │   *.npy
│   │   │   ...
    │
│   └───validationdepth
│       │   *.png
│       │   *.png
│       │   ...
│   
└───top
    │
    └───camera_params
    │   │   *.json
    │   │   *.json
    │   │   ...
    │
    └───train
    │   │   *.npy
    │   │   *.png
    │   │   ...
    │
    └───train2d
    │   │   *.npy
    │   │   *.npy
    │   │   ...
    │
    └───traindepth
    │   │   *.png
    │   │   *.png
    │   │   ...
    │
    └───validation
    │   │   *.npy
    │   │   *.png
    │   │   ...
    │
    └───validation2d
    │   │   *.npy
    │   │   *.npy
    │   │   ...
    │
    └───validationdepth
        │   *.png
        │   *.png
        │   ...
```
