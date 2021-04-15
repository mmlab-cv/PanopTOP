import open3d as o3d
import sys
import os

'''
for i in range(10, 29401, 10):
    print(i)
    input_file = f"./ptsclouds/ptcloud_hd{i:08}.ply"
    pcd = o3d.io.read_point_cloud(input_file)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=0.01)
    o3d.io.write_point_cloud(f"./trans/{i:08}.ply", pcd, write_ascii=True)
    os.remove(input_file)

'''
bad = list(range(10, 281, 10))+ \
        list(range(4360, 4861, 10)) + \
        list(range(8920, 9661, 10)) + \
        list(range(13660, 14121, 10)) + \
        list(range(18160, 18921, 10)) + \
        list(range(22900, 23631, 10)) + \
        list(range(27650, 29401, 10)) + [420]

'''
mypath = "./IMG2"

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file in onlyfiles:
        num = int(file[:8])

        if num in bad:
                os.remove(f"{mypath}/{file}")
'''

i = int(sys.argv[1])

if i in bad:
    quit(0)

# full rot: 2000
view = int(sys.argv[2])
rot = 500 * (view - 1)

print("i:", i)

pcd = o3d.io.read_point_cloud(f"/home/brodkap/Desktop/trans/{i:08}.ply")

# pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=0.01)

vis = o3d.visualization.Visualizer()

vis.create_window()

vis.add_geometry(pcd)

ctr = vis.get_view_control()
ctr.change_field_of_view(step=100)
ctr.rotate(rot, -550.0, 200, 200)
ctr.translate(-100, 100)

ctr.scale(3)

vis.poll_events()
vis.update_renderer()


vis.capture_screen_image(f"./IMG3/{i:08}_{view}.png")
vis.destroy_window()

# o3d.visualization.draw_geometries([pcd])
