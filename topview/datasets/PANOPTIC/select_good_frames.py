import sys
import os
import re
import json
import numpy as np
import gzip
from shutil import copyfile

good_frames_dict = {}

good_frames_dict[1] = [
    (290, 4350),
    (4870, 8910),
    (9670, 13650),
    (14130, 18150),
    (18930, 22890),
    (23640, 27640)
]

good_frames_dict[2] = [
    (300, 4000),
    (4800, 8200),
    (9600, 11350),
    #(14550, 18200),
    #(19580, 23000),
    #(24200, 28000),
    #(28850, 32100),
    #(33700, 37150)
]

good_frames_dict[3] = [
    (290, 4280),
    (5180, 8900)
]

good_frames_dict[4] = [
    (400, 4200),
    (4800, 8300),
    (9300, 12700),
    (13800, 17300),
    (18530, 21900),
    (22970, 26350),
    (27800, 31770)
]

good_frames_dict[5] = [
    (250, 4000),
    (4860, 8920),
    (9420, 13460),
    (14030, 17450),
    (18600, 22000),
    (23400, 26710),
]

def frames_bounds_to_list_of_record_num(bounds):
    ret = {}

    for i, (start, end) in enumerate(bounds):
        for el in range(start, end+1, 10):
            ret[el] = i

    return ret

def int_to_8digit(n):
    return f"{n:08}"

def generate_target_file(n_recording, person, num, extension):
    offsets = {
        1:0,
        2:6,
        3:9,
        4:11,
        5:18
    }

    num = (num - good_frames_dict[n_recording][person][0]) // 10
    person += offsets[n_recording]

    return f"{person:02}_{num:08}.{extension}"

def get_joint_file_name(n):
    return f"body3DScene_{int_to_8digit(n)}.json"

def get_cloud_file_name(n):
    return f"ptcloud_hd{int_to_8digit(n)}.ply"


n_recording = int(sys.argv[1])
frames_bounds = good_frames_dict[n_recording]


source_dir = f"/home/Datasets/PANOPTIC/raw/0{n_recording}/"
target_dir = f"/home/Datasets/PANOPTIC/"

source_joints_dir = source_dir + "joints"
source_clouds_dir = source_dir + "kinoptic_ptclouds"
target_joints_dir = target_dir + "joints"
target_clouds_dir = target_dir + "clouds"

frames = frames_bounds_to_list_of_record_num(frames_bounds)

for frame_num, person in frames.items():
    old_joint_file_name = f"{source_joints_dir}/{get_joint_file_name(frame_num)}"
    old_cloud_file_name = f"{source_clouds_dir}/{get_cloud_file_name(frame_num)}"
    print(old_cloud_file_name)


    new_joint_file_name = f"{target_joints_dir}/{generate_target_file(n_recording, person, frame_num, 'npy.gz')}"
    new_cloud_file_name = f"{target_clouds_dir}/{generate_target_file(n_recording, person, frame_num, 'ply')}"

    '''
    print(joint_target_file)
    print(cloud_target_file)
    print(new_joint_file_name)
    print(new_cloud_file_name)
    '''

    with open(old_joint_file_name) as json_file:
        data = json.load(json_file)
        arr_joints = np.array(data["bodies"][0]["joints19"]).reshape(-1, 4)

        f = gzip.GzipFile(new_joint_file_name, "w")
        np.save(file=f, arr=arr_joints)
        f.close()

    copyfile(old_cloud_file_name, new_cloud_file_name)