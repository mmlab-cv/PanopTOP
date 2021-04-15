import numpy as np
import pandas as pd
import gzip
from common import common_variables

DATA_DIR = f"{common_variables.DATA_PATH}/PANOPTIC/01"
indices_file = f"{DATA_DIR}/indices.py"

def get_joints_for_index(index):
    path = f"{DATA_DIR}/joints_positions/body3DScene_{index:08}.json"
    json = pd.read_json(path)

    joints = np.array(json["bodies"][0]["joints19"]) / 100
    joints = joints.reshape(-1, 4)[:, :3]

    x = joints[:, 0]
    y = joints[:, 2]
    z = 2.5 + joints[:, 1]

    joints = np.column_stack((x, y, z))

    f = gzip.GzipFile(f"{DATA_DIR}/joints/{index:08}.npy.gz", "w")
    np.save(file=f, arr=joints)
    f.close()


def get_point_cloud_for_index(index):
    path = f"{DATA_DIR}/trans/{index:08}.ply"
    arr = np.loadtxt(path)
    arr = arr[:, :3] / 100

    x = arr[:, 0].reshape(-1, 1)
    y = arr[:, 2].reshape(-1, 1)
    z = 2.5 + arr[:, 1].reshape(-1, 1)

    cp = np.concatenate((x, y, z), axis=1)

    f = gzip.GzipFile(f"{DATA_DIR}/point_cloud/{index:08}.npy.gz", "w")
    np.save(file=f, arr=cp)
    f.close()



    return cp

exec(open(indices_file).read())
# good_indices are taken from indices_file
for ind in good_indices:
    get_joints_for_index(ind)
    get_point_cloud_for_index(ind)


'''
def read_panoptic():
    centers = np.loadtxt("../centers_panoptic/centers_ITOP.txt")

    good_indices = np.where(np.abs(np.sum(centers, axis=1)) > 0.0001)[0]
    good_indices = [ind for ind in good_indices if ind % 50 == 0]
    np.random.shuffle(good_indices)

    train_ind, val_ind, test_ind = np.split(good_indices, [int(0.7 * len(good_indices)), int(0.85 * len(good_indices))])

    centers_train = centers[train_ind]
    centers_val = centers[val_ind]
    centers_test = centers[test_ind]

    centers = [centers_train, centers_val, centers_test]

    joints_train = []
    joints_val = []
    joints_test = []

    points_train = []
    points_val = []
    points_test = []

    for ind in train_ind:
        joints_train.append(get_joints_for_index(ind))
        points_train.append(get_point_cloud_for_index(ind))

    for ind in val_ind:
        joints_val.append(get_joints_for_index(ind))
        points_val.append(get_point_cloud_for_index(ind))
    for ind in test_ind:
        joints_test.append(get_joints_for_index(ind))
        points_test.append(get_point_cloud_for_index(ind))

    joints_train = np.array(joints_train)
    joints_val = np.array(joints_val)
    joints_test = np.array(joints_test)
    points_train = np.array(points_train)
    points_val = np.array(points_val)
    points_test = np.array(points_test)

    joints = [joints_train, joints_val, joints_test]
    points = [points_train, points_val, points_test]

    return points, centers, joints
'''