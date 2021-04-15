"""
=============================
Written by: Giulia Martinelli
=============================
==================
About this Script:
==================

This script file plot the changing of point clound and joints position with respect frames. 
The plot is give in 4 different views:
    - Free View
    - Side View
    - Top View
    - Front View

"""
import numpy as np
import open3d as o3d
import sys
import gzip
from pathlib import Path
import os
from collections import Counter
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd
import matplotlib.animation as animation
from matplotlib.pyplot import figure
import argparse

panoptic_connections = [[0,1],[0,2],[0,3],[3,4],[4,5],[2,6],[6,7],[7,8],[0,9],[9,10],[10,11],[2,12],[12,13],[13,14],[1,15],[1,16],[15,17],[16,18]]
path = (Path(__file__) / "/home/Datasets/PANOPTIC/").resolve()
joints_path = "/home/Datasets/PANOPTIC/joints/"

cloud_path = "/home/Datasets/PANOPTIC/clouds/"

def connectpoints(x,y,z,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    z1,z2 = z[p1],z[p2]
    return x1,x2,y1,y2,z1,z2

ap = argparse.ArgumentParser()
ap.add_argument('--ids', default=None, help='Person ID')
args = vars(ap.parse_args())


for (path, dirnames, filenames) in os.walk(joints_path):
    locals()["joints_files"+args["ids"]] = [os.path.join(path, name) for name in sorted(filenames) if name.startswith(args["ids"])]

print("File Joints Saved")

# for ids in subject_chosen:
for (path, dirnames, filenames) in os.walk(cloud_path):
    locals()["clouds_files"+args["ids"]] = [os.path.join(path, name) for name in sorted(filenames) if name.startswith(args["ids"])]
locals()["xcloud"+args["ids"]] = []
locals()["ycloud"+args["ids"]] = []
locals()["zcloud"+args["ids"]] = []
for i in range(len(locals()["clouds_files"+args["ids"]])):
    pcd = o3d.io.read_point_cloud(locals()["clouds_files"+args["ids"]][i])
    data = np.asarray(pcd.points)
    col = np.asarray(pcd.colors)
    locals()["xcloud"+args["ids"]].append(data[:,0]/100)
    locals()["ycloud"+args["ids"]].append(data[:,2]/100)
    locals()["zcloud"+args["ids"]].append(-data[:,1]/100)

print("File Clouds Saved")
print("Start")
plt.ion()
fig = plt.figure(figsize=(10,10))
# for ids in subject_chosen:
for num in range(len(locals()["xcloud"+args["ids"]])):
    print(locals()["joints_files"+args["ids"]][num])
    data = np.load(gzip.GzipFile(locals()["joints_files"+args["ids"]][num], "r"))
    data = data[:,:3]
    xdata = data[:,0]/100
    ydata = data[:,2]/100
    zdata = -data[:,1]/100
    ax0= fig.add_subplot(2, 2, 1, projection='3d')
    ax0.scatter(locals()["xcloud"+args["ids"]][num], locals()["ycloud"+args["ids"]][num], locals()["zcloud"+args["ids"]][num],c = '#009aff3a', alpha = 0.01, marker='o')
    ax0.scatter(xdata, ydata, zdata,c ='#FF0000', marker = 'o')
    for i in panoptic_connections:
        x1,x2,y1,y2, z1,z2 = connectpoints(xdata, ydata, zdata, i[0],i[1])
        ax0.plot([x1,x2],[y1,y2],[z1,z2],'r-')
    ax0.set_xlabel('x', rotation=0, fontsize=10, labelpad=5)
    ax0.set_ylabel('y', rotation=0, fontsize=10, labelpad=5)
    ax0.set_zlabel('z', rotation=0, fontsize=10, labelpad=5)
    ax0.set_xlim3d(-1,1)
    ax0.set_ylim3d(-2,2)
    ax0.set_zlim3d(0,2)
    ax0.set_title('Free View', size=15)

    ax1= fig.add_subplot(2, 2, 2, projection='3d')
    ax1.scatter(locals()["xcloud"+args["ids"]][num], locals()["ycloud"+args["ids"]][num], locals()["zcloud"+args["ids"]][num], c = '#009aff3a', alpha = 0.01, marker='o')
    ax1.scatter(xdata, ydata, zdata,c ='#FF0000', marker = 'o')
    for i in panoptic_connections:
        x1,x2,y1,y2, z1,z2 = connectpoints(xdata, ydata, zdata, i[0],i[1])
        ax1.plot([x1,x2],[y1,y2],[z1,z2],'r-')
    ax1.view_init(elev=0., azim=-90)
    ax1.set_xlabel('x', rotation=0, fontsize=10, labelpad=5)
    ax1.set_ylabel('y', rotation=0, fontsize=10, labelpad=5)
    ax1.set_zlabel('z', rotation=0, fontsize=10, labelpad=5)
    ax1.set_xlim3d(-1,1)
    ax1.set_ylim3d(-2,2)
    ax1.set_zlim3d(0,2)
    ax1.set_title('Side View',size = 15)


    ax2= fig.add_subplot(2, 2, 3, projection='3d')
    ax2.scatter(locals()["xcloud"+args["ids"]][num], locals()["ycloud"+args["ids"]][num], locals()["zcloud"+args["ids"]][num], c = '#009aff3a', alpha = 0.01, marker='o')
    ax2.scatter(xdata, ydata, zdata,c ='#FF0000', marker = 'o')
    for i in panoptic_connections:
        x1,x2,y1,y2, z1,z2 = connectpoints(xdata, ydata, zdata, i[0],i[1])
        ax2.plot([x1,x2],[y1,y2],[z1,z2],'r-')
    ax2.view_init(elev=90., azim=180)
    ax2.set_xlabel('x', rotation=0, fontsize=10, labelpad=5)
    ax2.set_ylabel('y', rotation=0, fontsize=10, labelpad=5)
    ax2.set_zlabel('z', rotation=0, fontsize=10, labelpad=5)
    ax2.set_xlim3d(-1,1)
    ax2.set_ylim3d(-2,2)
    ax2.set_zlim3d(0,2)
    ax2.set_title('Top View',size = 15)


    ax3= fig.add_subplot(2, 2, 4, projection='3d')
    ax3.scatter(locals()["xcloud"+args["ids"]][num], locals()["ycloud"+args["ids"]][num], locals()["zcloud"+args["ids"]][num], c = '#009aff3a', alpha = 0.01, marker='o')
    ax3.scatter(xdata, ydata, zdata,c ='#FF0000', marker = 'o')
    for i in panoptic_connections:
        x1,x2,y1,y2, z1,z2 = connectpoints(xdata, ydata, zdata, i[0],i[1])
        ax3.plot([x1,x2],[y1,y2],[z1,z2],'r-')
    ax3.view_init(elev=0., azim=180)
    ax3.set_xlabel('x', rotation=0, fontsize=10, labelpad=5)
    ax3.set_ylabel('y', rotation=0, fontsize=10, labelpad=5)
    ax3.set_zlabel('z', rotation=0, fontsize=10, labelpad=5)
    ax3.set_xlim3d(-1,1)
    ax3.set_ylim3d(-2,2)
    ax3.set_zlim3d(0,2)
    ax3.set_title('Front View', fontsize = 15)
    folder_path = "/home/giuliamartinelli/SPIN/Top-View/results/" + args["ids"]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fig.suptitle('ID{ID} Frame_{number}'.format(ID = args["ids"], number = num), fontsize = 25)
    path = os.path.join(folder_path,"frame_{0}.png".format(num))
    plt.savefig(path)
    print('Done ID{ID} Frame_{number}'.format(ID = args["ids"], number = num))
    ax0.cla()
    ax1.cla()
    ax2.cla()
    ax3.cla()
        # plt.close
