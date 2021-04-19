# ICIP-2021-2346
Anonymous dataset and code for **ICIP 2021 paper #2346**

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