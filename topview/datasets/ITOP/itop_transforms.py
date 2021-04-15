import h5py
import numpy as np
import os
import gzip
import sys

sys.path.append("../../..")
from topview.common import get_big_data_path


def openh5(cloud_path, label_path):
    cloud_file = h5py.File(cloud_path, 'r')
    cloud_ids, clouds = cloud_file.get('id'), cloud_file.get('data')
    cloud_ids, clouds = np.asarray(cloud_ids), np.asarray(clouds)

    label_file = h5py.File(label_path, 'r')
    label_ids, is_valid, real_world_coordinates, segmentation = \
        label_file.get('id'), label_file.get('is_valid'), label_file.get('real_world_coordinates'), label_file.get('segmentation')
    label_ids, is_valid, real_world_coordinates, segmentation = \
        np.asarray(label_ids), np.asarray(is_valid), np.asarray(real_world_coordinates), np.asarray(segmentation),

    def byte_arr_to_str_arr(byte_arr):
        return [el.decode("utf-8") for el in byte_arr]
    
    cloud_ids, label_ids = byte_arr_to_str_arr(cloud_ids), byte_arr_to_str_arr(label_ids)
     
    # check if all elements are equal
    assert(cloud_ids == label_ids)


    arrays = [label_ids, is_valid, real_world_coordinates, segmentation, clouds]

    frames = [
        {
            "id": x[0], 
            "is_valid": x[1], 
            "real_world_coordinates": x[2],
            "segmentation": x[3],
            "cloud": x[4]
        } for x in zip(*arrays)
    ]

    frames = list(filter(lambda frame: frame["is_valid"] == 1, frames))
    
    def transform_id_to_correct_format(frame):
        frame["id"] = frame["id"][:3] + "00" + frame["id"][3:]
        return frame

    frames = list(map(transform_id_to_correct_format, frames))

    def filter_point_clouds(frame):
        '''Only retained those point clouds, that are ok after segmentation'''
        body_coords = np.where(frame["segmentation"] != -1)
        frame["cloud"] = frame["cloud"].reshape(240, 320, 3)[body_coords].reshape(-1, 3)
        return frame

    frames = list(map(filter_point_clouds, frames))

    return frames


# train: 16 people; test: 4 people
# mode: train / test
def itop_file_path(data_dir, viewpoint, mode, type):
    return f'{data_dir}/raw/ITOP_{viewpoint}_{mode}_{type}.h5'

def read_itop(viewpoint, mode):
    DATA_DIR = f"{get_big_data_path()}/ITOP"

    cloud_path = itop_file_path(DATA_DIR, viewpoint, mode, "point_cloud")
    label_path = itop_file_path(DATA_DIR, viewpoint, mode, "labels")

    frames = openh5(cloud_path, label_path)
    print("read_itop", viewpoint)


    def get_frame_person_id(frame):
        return frame["id"][:2]

    print("person ids", set(map(get_frame_person_id, frames)))

    # at the beginning those were indices of people from train / test 
    # {'06', '09', '10', '07', '19', '18', '12', '08', '11', '15', '16', '17', '04', '05', '13', '14'}
    # {'00', '03', '01', '02'}
    # following changes chaged it to
    # {'00' - '15'}
    # {'16' - '19'}
    
    def transform_test_id(frame):
        frame_id = frame["id"]
        frame_id = str(int(frame_id[:2]) + 16) + frame_id[2:]
        frame["id"] = frame_id
        return frame

    def transform_train_id(frame):
        frame_id = frame["id"]
        frame_id = f"{(int(frame_id[:2]) - 4):02d}{frame_id[2:]}"
        frame["id"] = frame_id
        return frame 

    if mode == "test":
        frames = list(map(transform_test_id, frames))
    elif mode == "train":
        frames = list(map(transform_train_id, frames))

    
    return frames

#train_frames = read_itop("top", "train")
#test_frames = read_itop("top", "test")

train_frames = read_itop("side", "train")
test_frames = read_itop("side", "test")

for frame in train_frames + test_frames:
    cloud = frame["cloud"]
    joints = frame["real_world_coordinates"]
    frame_id = frame["id"]
    
    DATA_DIR = f"{get_big_data_path()}/ITOP"
    #cloud_path = f"{DATA_DIR}/clouds_depth/{frame_id}.npy.gz"
    #joints_path = f"{DATA_DIR}/joints_side/{frame_id}.npy.gz"
    cloud_path = f"{DATA_DIR}/clouds_depth_side/{frame_id}.npy.gz"
    joints_path = f"{DATA_DIR}/joints_side/{frame_id}.npy.gz"

    print("cloud_path, joints_path", cloud_path, joints_path)

    f = gzip.GzipFile(cloud_path, "w")
    np.save(file=f, arr=cloud)
    f.close()

    f = gzip.GzipFile(joints_path, "w")
    np.save(file=f, arr=joints)
    f.close()
