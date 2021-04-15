# %%
import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import gzip

sys.path.append("../../..")
from topview.common import get_big_data_path

JOINTS3D =  f"{get_big_data_path()}/ITOP/joints_side"
JOINTS2D =  f"{get_big_data_path()}/ITOP/for_capsules/side/joints_2d"
DIM = f"{get_big_data_path()}/ITOP/for_capsules/side/depths"	

# %%
for fname in os.listdir(JOINTS2D):
    os.rename(f"{JOINTS2D}/{fname}", f"{JOINTS2D}/{fname.replace('_side', '')}")

# %%
fnames3d = os.listdir(JOINTS3D)
fnames2d = os.listdir(JOINTS2D)

# %%
common_fnames = list(set(fnames3d) & set(fnames2d))
print(len(common_fnames))

target = f"{get_big_data_path()}/ITOP/for_capsules/side/joints_3dand2d"
# %%
def get_joints_itop(path):
    f = gzip.GzipFile(path, "r")
    return np.load(f)

for fname in common_fnames:
    path_3d = f"{JOINTS3D}/{fname}"
    path_2d = f"{JOINTS2D}/{fname}"

    arr3 = get_joints_itop(path_3d)
    arr2 = get_joints_itop(path_2d)

    arr5 = np.concatenate((arr3, arr2), axis=1)

    target_path = f"{target}/{fname}"
    f = gzip.GzipFile(target_path, "w")
    np.save(file=f, arr=arr5)
    f.close()
# %%
