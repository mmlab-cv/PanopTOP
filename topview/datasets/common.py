import numpy as np
import gzip
from plyfile import PlyData

def get_center_of_point_cloud(cloud):
    return np.mean(cloud, axis=0)


def save_arr_to_gzip(arr_tp_save, path):
    f = gzip.GzipFile(path, "w")
    np.save(file=f, arr=arr_to_save)
    f.close()


def get_arr_from_itop_gzip(path):
    f = gzip.GzipFile(path, "r")
    return np.load(f)

def get_arr_from_ply_cloud(path):
    plydata = PlyData.read(path)
    vertex_data = plydata['vertex'].data
    pts = np.zeros([vertex_data.size, 3])
    pts[:, 0] = vertex_data['z']
    pts[:, 1] = vertex_data['x']
    pts[:, 2] = 250 + vertex_data['y']
    pts = pts / 100
    return pts

def get_arr_from_panoptic_gzip(path):
    f = gzip.GzipFile(path, "r")
    tmp_arr = np.load(f)
    jts = np.zeros((tmp_arr.shape[0], 3))

    jts[:, 0] = tmp_arr[:, 2]
    jts[:, 1] = tmp_arr[:, 0]
    jts[:, 2] = 250 + tmp_arr[:, 1]
    jts = jts / 100
    return jts

def get_cloud_itop(path):
    f = gzip.GzipFile(path, "r")
    return np.load(f)

def get_joints_itop(path):
    f = gzip.GzipFile(path, "r")
    return np.load(f)


def change_format_from_19_joints_to_15_joints(joints):
    xdata = joints[:,0]
    ydata = joints[:,1]
    zdata = joints[:,2]

    panoptic_head = [(xdata[16]+xdata[18])/2,(ydata[16]+ydata[18])/2,(zdata[16]+zdata[18])/2]
    panoptic_torso = [(xdata[0]+xdata[2])/2,(ydata[0]+ydata[2])/2,(zdata[0]+zdata[2])/2]


    #                           head        neck      r shoulder l shoulder r elbow  l elbow     r hand    l hand      torso            r hip       l hip    r knee     l knee    r foot    l foot
    #xdata_new = np.array([panoptic_head[0], xdata[0], xdata[9], xdata[3], xdata[10], xdata[4], xdata[11], xdata[5], panoptic_torso[0], xdata[12], xdata[6], xdata[13], xdata[7], xdata[14], xdata[8]])
    #ydata_new = np.array([panoptic_head[1], ydata[0], ydata[9], ydata[3], ydata[10], ydata[4], ydata[11], ydata[5], panoptic_torso[1], ydata[12], ydata[6], ydata[13], ydata[7], ydata[14], ydata[8]])
    #zdata_new = np.array([panoptic_head[2], zdata[0], zdata[9], zdata[3], zdata[10], zdata[4], zdata[11], zdata[5], panoptic_torso[2], zdata[12], zdata[6], zdata[13], zdata[7], zdata[14], zdata[8]])

    xdata_new = np.array([panoptic_head[0], xdata[0], xdata[3], xdata[9], xdata[4], xdata[10], xdata[5], xdata[11], panoptic_torso[0], xdata[6], xdata[12], xdata[7], xdata[13], xdata[8], xdata[14]])
    ydata_new = np.array([panoptic_head[1], ydata[0], ydata[3], ydata[9], ydata[4], ydata[10], ydata[5], ydata[11], panoptic_torso[1], ydata[6], ydata[12], ydata[7], ydata[13], ydata[8], ydata[14]])
    zdata_new = np.array([panoptic_head[2], zdata[0], zdata[3], zdata[9], zdata[4], zdata[10], zdata[5], zdata[11], panoptic_torso[2], zdata[6], zdata[12], zdata[7], zdata[13], zdata[8], zdata[14]])

    panoptic_converted = np.empty(shape=(15, 3), dtype=float)
    for index in range(len(panoptic_converted)):
        panoptic_converted[index,0] = xdata_new[index]
        panoptic_converted[index,1] = ydata_new[index]
        panoptic_converted[index,2] = zdata_new[index]

    return panoptic_converted


def get_cloud_panoptic(path):
    plydata = PlyData.read(path)
    vertex_data = plydata['vertex'].data
    pts = np.zeros([vertex_data.size, 3])
    pts[:, 0] = vertex_data['z']
    pts[:, 1] = vertex_data['x']
    pts[:, 2] = 250 + vertex_data['y']
    pts = pts / 100
    return pts

def get_joints_panoptic(path):
    f = gzip.GzipFile(path, "r")
    tmp_arr = np.load(f)
    jts = np.zeros((tmp_arr.shape[0], 3))

    jts[:, 0] = tmp_arr[:, 2]
    jts[:, 1] = tmp_arr[:, 0]
    jts[:, 2] = 250 + tmp_arr[:, 1]
    jts = jts / 100
    return change_format_from_19_joints_to_15_joints(jts)

