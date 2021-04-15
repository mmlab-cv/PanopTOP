import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

sys.path.append("../../")
from topview.common import get_big_data_path
from topview.datasets.common import get_cloud_itop, get_joints_itop, get_cloud_panoptic, get_joints_panoptic

ITOP = "ITOP"
PANOPTIC = "PANOPTIC"
BOTH = "BOTH"

CLOUD = "cloud"
JOINTS = "joints"
DEPTH_IMAGES = "depth_images"
JOINTS2D = "2Djoints"
JOINTS3DAND2D = "joints3Dand2D"
SIDE_CLOUD = "side_cloud"
SIDE_JOINTS = "side_joints"
SIDE_DEPTH_IMAGES = "side_depth_images"
SIDE_JOINTS2D = "side_2Djoints"
SIDE_JOINTS3DAND2D = "side_joints3Dand2D"

class DBDataset():
    def __init__(self, name, fraction, x_type = CLOUD, y_type = JOINTS):
        assert(name in [ITOP, PANOPTIC, BOTH])
        assert(fraction in ["train", "validation", "test"])
        self.x_type = x_type
        self.y_type = y_type

        file_directory = {
            ITOP: {
                CLOUD: f"{get_big_data_path()}/ITOP/clouds_depth",
                JOINTS: f"{get_big_data_path()}/ITOP/joints",
                DEPTH_IMAGES: f"{get_big_data_path()}/ITOP/for_capsules/top/depths",
                JOINTS2D: f"{get_big_data_path()}/ITOP/for_capsules/top/joints_2d",
                JOINTS3DAND2D: f"{get_big_data_path()}/ITOP/for_capsules/top/joints_3dand2d",
                SIDE_CLOUD: f"{get_big_data_path()}/ITOP/clouds_depth_side",
                SIDE_JOINTS: f"{get_big_data_path()}/ITOP/joints_side",
                SIDE_DEPTH_IMAGES: f"{get_big_data_path()}/ITOP/for_capsules/side/depths",
                SIDE_JOINTS2D: f"{get_big_data_path()}/ITOP/for_capsules/side/joints_2d",
                SIDE_JOINTS3DAND2D: f"{get_big_data_path()}/ITOP/for_capsules/side/joints_3dand2d"
            },
            PANOPTIC: {
                # CLOUD: f"{get_big_data_path()}/PANOPTIC/clouds_top",
                # JOINTS: f"{get_big_data_path()}/PANOPTIC/joints"
            }
        }

        file_names = {k_outer: {k_inner: os.listdir(v_inner) for k_inner, v_inner in dict_outer.items()} \
            for k_outer, dict_outer in file_directory.items()}
        
        for k, v in file_names.items():
            print(k)
            for k2, v2 in v.items():
                print(k2, v2[:3])


        def select(cloud_file_names, joint_file_names, dataset_name):
            pref_cloud = list(map(lambda x: x[:11], cloud_file_names))
            pref_joint = list(map(lambda x: x[:11], joint_file_names))
            
            suff_cloud = cloud_file_names[0][11:]
            suff_joint = joint_file_names[0][11:]
            # from pdb import set_trace as bp
            # bp()
            common_pref = list(set(pref_cloud) & set(pref_joint))

            def get_correct(prefixes):
                num_frames_per_user = {
                    ITOP: 420,
                    PANOPTIC: 350
                }

                fraction_boundaries = {
                    ITOP: {
                        "train": (4, 19), 
                        "validation": (0, 3),
                        "test": (0, 3)
                    },
                    PANOPTIC: {
                        "train": (0, 17), 
                        "validation": (19, 20),
                        "test": (21, 23)
                    }
                }

                user_to_fnames = {}
                for file_prefix in prefixes:
                    user = int(file_prefix[:2])
                    if user not in user_to_fnames:
                        user_to_fnames[user] = [file_prefix]
                    else:
                        user_to_fnames[user].append(file_prefix)
                
                if dataset_name == ITOP:
                    user_to_fnames = {k: random.sample(v, num_frames_per_user[ITOP]) \
                        for k, v  in user_to_fnames.items()}
                elif dataset_name == PANOPTIC:
                    user_to_fnames = {k: v[:330] for k, v in user_to_fnames.items()}

                users_boundaries = fraction_boundaries[dataset_name][fraction]
                ret_prefixes = []
                for i in range(users_boundaries[0], users_boundaries[1] + 1):
                    ret_prefixes += user_to_fnames[i]

                return ret_prefixes

            common_pref = get_correct(common_pref)
            return [pref + suff_cloud for pref in common_pref], [pref + suff_joint for pref in common_pref]


        if name in [ITOP, BOTH]:
            file_names[ITOP][x_type], file_names[ITOP][y_type] = \
                select(file_names[ITOP][x_type], file_names[ITOP][y_type], ITOP)

        if name in [PANOPTIC, BOTH]:
            file_names[PANOPTIC][x_type], file_names[PANOPTIC][y_type] = \
                select(file_names[PANOPTIC][x_type], file_names[PANOPTIC][y_type], PANOPTIC)

        file_paths = {dataset_key: \
            {dataset_type: [f"{file_directory[dataset_key][dataset_type]}/{fname}" for fname in file_names[dataset_key][dataset_type]] \
            for dataset_type in file_directory[dataset_key].keys()} for dataset_key in [ITOP, PANOPTIC]}

        if name == BOTH:
            self.X_paths = file_paths[ITOP][x_type] + file_paths[PANOPTIC][x_type]
            self.Y_paths = file_paths[ITOP][y_type] + file_paths[PANOPTIC][y_type]
        else:
            self.X_paths = file_paths[name][x_type]
            self.Y_paths = file_paths[name][y_type]


    def get_dataset(self):
        X_data = []
        Y_data = []

        for X_path, Y_path in zip(self.X_paths, self.Y_paths):
            if ITOP in X_path:
                # from pdb import set_trace as bp
                # bp()
                if self.x_type == CLOUD or self.x_type == SIDE_CLOUD:
                    X_element = get_cloud_itop(X_path)
                elif self.x_type == DEPTH_IMAGES or self.x_type == SIDE_DEPTH_IMAGES:
                    X_element = plt.imread(X_path)

                Y_element = get_joints_itop(Y_path)
                
            elif PANOPTIC in X_path:
                if self.x_type == CLOUD or self.x_type == SIDE_CLOUD:
                    X_element = get_cloud_panoptic(X_path)
                elif self.x_type == DEPTH_IMAGES or self.x_type == SIDE_DEPTH_IMAGES:
                    X_element = plt.imread(X_path)

                Y_element = get_joints_panoptic(Y_path)

            X_data.append(X_element)
            Y_data.append(Y_element)

        return X_data, Y_data, self.X_paths, self.Y_paths


# test the scipt
'''
import numpy as np

PANOPTIC = DBDataset("PANOPTIC", "train")
cl, jts, _, _ = PANOPTIC.get_dataset()
np.save("tmp/pan_cl", cl[0])
np.save("tmp/pan_jt", jts[0])

ITOP_set = DBDataset("ITOP", "train", "depth_images", "2Djoints")
cl, jts, _, _ = ITOP_set.get_dataset()
print(len(cl), len(jts))
'''
