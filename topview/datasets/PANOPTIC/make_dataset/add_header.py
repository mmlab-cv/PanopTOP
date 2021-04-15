import os

ply_folder = "/home/Datasets/PANOPTIC/raw/01/point_cloud_copy"
out_folder = "/home/Datasets/PANOPTIC/raw/01/point_cloud_w_header"

prefix_1 = '''ply
format ascii 1.0
element vertex''' 

prefix_2 = '''property double x
property double y
property double z
property uchar red
property uchar green
property uchar blue
end_header
'''

for cloud in os.listdir(ply_folder):
    cloud_path = os.path.join(ply_folder,cloud)
    num_lines = sum(1 for line in open(cloud_path))
    with open(cloud_path) as f, open(os.path.join(out_folder,f"{cloud}"), "w") as out:
        out.write(prefix_1)
        out.write(" "+str(num_lines)+"\n")
        out.write(prefix_2)
        for line in f:
            out.write(line)

