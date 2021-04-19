# Datasets
1. ITOP (https://zenodo.org/record/3932973#.X3M-Wmgza70)
    * depth: point of clouds: set of points (x, y, z)
    * joints positions: 15 joints with position (x, y, z)
2. TVPR (https://vrai.dii.univpm.it/re-id-dataset)
    * RGB images
    * depth: point of clouds: set of points (x, y, z)
3. PANOPTIC:
    * RGB images (https://www.cs.cmu.edu/~hanbyulj/panoptic-studio/)
    * depth: point of clouds: set of points (x, y, z)
    * joints positions: 19 joints with position (x, y, z)


Originally, datasets come in different forms, in `topview/datasets` directory there are scripts for transforming them into final form. The final form is described in form of contract:
* clouds of points are saved in **{person}_{num}.npy.gz** numpy file as a (n, 3) dataframe,
* positions of joints are saved in **{person}_{num}.npy.gz** numpy file as a (1, 3 * 15) or (1, 3 * 19) dataframe

**{person}** - person id, 2 digit number starting from 00,

**{num}** - number of frame per person is a natural number taking 8 digits starting from 00000000 **WARNING** numbers are not necessarily consecutive. There is a list to check it when loading (TODO: write a function that takes care for that for all datasets).

In **external_volume** there are 3 folders, each per dataset:
* ITOP
* TVPR
* PANOPTIC


In those folders there are folders (if they apply):
* `raw` - data in its original form, as downloaded from provider, 
* `joints` - .npy.gz
* `clouds` - .ply files
* `clouds_depth` - .npy.gz
* `images_depth` - .npy.gz
* `clouds_rgb` - .png
* `images_rgb` - .png

### ITOP dataset
Originally, the dataset is stored in .h5 files. Transformation to desired form is in script `itop_transforms.py`.

Folders:
* **raw**
* **joints**
* **clouds_depth**

People from train and test are not intersecting.




### TVPR dataset (todo)

### PANOPTIC dataset (todo)
* panoptic-toolbox
* PANOPTIC




#### Precomputed voxelization
Because voxelization is a process needed when training and testing, it is proprocessed to save the time for different tes


## Panoptic processing
We will be using scripts from toolbox:
`https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox`
This toolboc is cloned to external memory to directory `panoptic-toolbox`

#### Downloading

Currently, following videos are downloaded:
* 171204_pose1
* 171204_pose2



../MatlabR2019b/bin/matlab -nosplash -nodesktop -r "run matlab/kinoptic_second.m"

#### How to download and preprocess to cloud of points:
1. Downloading:
    - `scripts/getData.sh 171204_poseX` (this script is modified in relation to original one; no need to download not needed data from not kinect cameras; only downloads joints positions)
    - `scripts/getData_kinoptic.sh 171204_pose1`
2. Extract images from video with script `cd 171204_poseX; {path_to_parent_of_kinectImgsExtractor}/kinectImgsExtractor.sh`. We need to do it from inside of `171204_poseX`.
1. (obosolete) Extract images from video with script, e.g. `scripts/extractAll.sh scripts/171204_poseX`
3. Run matlab script `matlab/demo_kinoptic_gen_ptcloud.m` - you need to set frames, that you want to extract, I extracted cloud of points from every 10th frame.
4. Clouds need to be cleaned: Clouds have many distortions, outliers

Crop big image, add texture: default size of cropped image is 400x400.
Create depth image


## Visualization of skeletons
For debugging or visualization of results, we can visualize joints that were obtained:

* run Python script **results/visualize_joints.py**
* dependencies to be installed: open3d, numpy

We run script with argument: file with results. Expected input: txt file with 45 (15 joints) or 57 (19 joints) in each row.

We are navigating with the use of keyboard: A: previous image, D: next image

The viewpoint at the beginning is probably bad. Please manipulate the viewpoint and save it with pressing S. It will be saved to camera.json file. Close and open again. Press L to load the viewpoint, now it should work with set viewpoint.  

**WARNING**: don’t manipulate size of the window, loading of viewpoint will not work then. If you did (for example if display changed), make Save and Load again.