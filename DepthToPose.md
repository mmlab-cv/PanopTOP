# Depth to pose
For depth to pose author is using the following repo: https://github.com/dragonbook/V2V-PoseNet-pytorch
To run this code, go to `depth_to_pose` repo and run:

```console
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./ python pose/main.py -e 5 -s 
```

## Test
### Modalities
* Random
* Grouped by person

### Inside dataset testing and cross-dataset testing 
* Train on ITOP test on ITOP
* Train on ITOP test on PANOPTIC
* Train on PANOPTIC test on ITOP
* Train on PANOPTIC test on PANOPTIC

## Results:
### Random

### Grouped by person
* Train on ITOP test on ITOP
* Train on ITOP test on PANOPTIC
* Train on PANOPTIC test on ITOP
* Train on PANOPTIC test on PANOPTIC


**TODO** Test integral pose loss