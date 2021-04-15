import open3d as o3d
import os
import sys

dir = sys.argv[1]

for i in range(200, 10000, 10):
    file_pt_cloud = f'{dir}/ptcloud_hd{i:08}.ply'
    pcd = o3d.io.read_point_cloud(file_pt_cloud)

    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.1)

    o3d.io.write_point_cloud(f"{dir}/../removed_noise/{i:08}.ply", pcd, write_ascii=True)
