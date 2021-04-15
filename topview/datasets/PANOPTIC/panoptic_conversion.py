"""
=============================
Written by: Giulia Martinelli
=============================
==================
About this Script:
==================

This script file defines the conversion of skeleton extracted from PANOPTIC dataset into the ITOP format.
The script performs the following:

1. Plot one frame of the ITOP keypoints in its reference frame
2. Plot one frame of the PANOPTIC keypoints in its reference frame
3. Plot one frame of the PANOPTIC keypoints in ITOP reference frame
4. Plot one frame of the PANOPTIC keypoints with ITOP keypoints mapping

"""

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math as m
import natsort
import gzip
##----------- Definition of Rotation Matrix----------
def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
##-----------------------------------------------------
def connectpoints(x,y,z,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    z1,z2 = z[p1],z[p2]
    return x1,x2,y1,y2,z1,z2
##-----------------------------ITOP---------------------------------
itop_labels = ['Head','Neck','RShould','LShould',"RElbow","LElbow","RHand","LHand","Torso","RHip","LHip","RKnee","LKnee","RFoot","LFoot"]
itop_connections = [[0,1],[1,2],[1,3],[2,3],[2,4],[3,5],[4,6],[5,7],[1,8],[8,9],[8,10],[9,10],[9,11],[10,12],[11,13],[12,14]]
I = gzip.GzipFile('/home/Datasets/ITOP/joints/00_0000000.npy.gz', "r")
itop = np.load(I)
fig = plt.figure()
ax = plt.axes(projection='3d')
xdata = itop[:,0].flatten()
ydata = itop[:,1].flatten()
zdata = itop[:,2].flatten()

for i in itop_connections:
    x1,x2,y1,y2,z1,z2 = connectpoints(xdata,ydata,zdata,i[0],i[1])
    ax.plot([x1,x2],[y1,y2],[z1,z2],'k-')

ax.scatter3D(xdata, ydata, zdata, c=zdata)

for x, y, z, label in zip(xdata,ydata,zdata, itop_labels):
    ax.text(x, y, z, label)

ax.text2D(0.05, 0.95, "ITOP", transform=ax.transAxes)

ax.set_xlabel('x', rotation=0, fontsize=20, labelpad=20)
ax.set_ylabel('y', rotation=0, fontsize=20, labelpad=20)
ax.set_zlabel('z', rotation=0, fontsize=20, labelpad=20)
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-2,2)
ax.set_zlim3d(0,2)
plt.show(block=False)
input('press <ENTER> to continue')

#-----------------------------------------------------PANOPTIC--------------------------------------------------------------------
panoptic_labels = ['Neck',"Nose","HipsCenter","LShould","LElbow","LWrist","LHip","LKnee","LAnkle","RShould","RElbow","RWrist","RHip","RKnee","RAnkle","REye","REars","LEye","LEars"]
panoptic_connections = [[0,1],[0,2],[0,3],[3,4],[4,5],[2,6],[6,7],[7,8],[0,9],[9,10],[10,11],[2,12],[12,13],[13,14],[1,15],[1,16],[15,17],[16,18]]
P = gzip.GzipFile('/home/Datasets/PANOPTIC/joints/00_00000000.npy.gz', "r")
panoptic = np.load(P)
fig = plt.figure()
ax = plt.axes(projection='3d')
xdata = panoptic[:,0]/100
ydata = panoptic[:,2]/100
zdata = -panoptic[:,1]/100

for i in panoptic_connections:
    x1,x2,y1,y2,z1,z2 = connectpoints(xdata,ydata,zdata,i[0],i[1])
    ax.plot([x1,x2],[y1,y2],[z1,z2],'k-')

ax.scatter3D(xdata, ydata, zdata, c=zdata)

for x, y, z, label in zip(xdata,ydata,zdata, panoptic_labels):
    ax.text(x, y, z, label)

ax.text2D(0.05, 0.95, "PANOPTIC", transform=ax.transAxes)

ax.set_xlabel('x', rotation=0, fontsize=20, labelpad=20)
ax.set_ylabel('y', rotation=0, fontsize=20, labelpad=20)
ax.set_zlabel('z', rotation=0, fontsize=20, labelpad=20)
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-2,2)
ax.set_zlim3d(0,2)
plt.show(block=False)
input('press <ENTER> to continue')
#---------------------------PANOPTIC to ITOP reference frame allignment------------------------------

alpha = m.pi
R = Ry(-m.pi/2)*Rx(m.pi)
fig = plt.figure()
ax = plt.axes(projection='3d')
panoptic_new = np.empty(shape=(19, 3), dtype=float)
for index in range(len(panoptic)):
    panoptic_new[index] = panoptic[:,:3][index]*R
xdata = panoptic_new[:,0]/100
ydata = -panoptic_new[:,2]/100
zdata = -panoptic_new[:,1]/100+2.5
for i in panoptic_connections:
    x1,x2,y1,y2,z1,z2 = connectpoints(xdata,ydata,zdata,i[0],i[1])
    ax.plot([x1,x2],[y1,y2],[z1,z2],'k-')

ax.scatter3D(xdata, ydata, zdata, c=zdata)

for x, y, z, label in zip(xdata,ydata,zdata, panoptic_labels):
    ax.text(x, y, z, label)

ax.text2D(0.05, 0.95, "PANOPTIC to ITOP reference frame", transform=ax.transAxes)

ax.set_xlabel('x', rotation=0, fontsize=20, labelpad=20)
ax.set_ylabel('y', rotation=0, fontsize=20, labelpad=20)
ax.set_zlabel('z', rotation=0, fontsize=20, labelpad=20)
ax.set_xlim3d(-0.5,0.5)
ax.set_ylim3d(-2,2)
ax.set_zlim3d(0,2)

plt.show(block=False)
input('press <ENTER> to continue')

#-----------------------------------------------PANOPTIC to ITOP keypoints mapping----------------------------------------------------------------------------------------------------------------
'''
The PANOPTIC dataset has 19 joints, the ITOP dataset has 15 joints. We performed the mapping in this way:
    - Head joint is the median point of the ears;
    - Torso joint is the median point of the neck and hip center;
    - Hands joints = Wrist joints
    - Foot joints = Ankle joints
    
'''
fig = plt.figure()
ax = plt.axes(projection='3d')
panoptic_head = [(xdata[16]+xdata[18])/2,(ydata[16]+ydata[18])/2,(zdata[16]+zdata[18])/2]
panoptic_torso = [(xdata[0]+xdata[2])/2,(ydata[0]+ydata[2])/2,(zdata[0]+zdata[2])/2]

xdata_new = np.array([panoptic_head[0], xdata[0], xdata[9], xdata[3], xdata[10], xdata[4], xdata[11], xdata[5], panoptic_torso[0], xdata[12], xdata[6], xdata[13], xdata[7], xdata[14], xdata[8]])
ydata_new = np.array([panoptic_head[1], ydata[0], ydata[9], ydata[3], ydata[10], ydata[4], ydata[11], ydata[5], panoptic_torso[1], ydata[12], ydata[6], ydata[13], ydata[7], ydata[14], ydata[8]])
zdata_new = np.array([panoptic_head[2], zdata[0], zdata[9], zdata[3], zdata[10], zdata[4], zdata[11], zdata[5], panoptic_torso[2], zdata[12], zdata[6], zdata[13], zdata[7], zdata[14], zdata[8]])

for i in itop_connections:
    x1,x2,y1,y2,z1,z2 = connectpoints(xdata_new,ydata_new,zdata_new,i[0],i[1])
    ax.plot([x1,x2],[y1,y2],[z1,z2],'k-')

ax.scatter3D(xdata_new, ydata_new, zdata_new, c=zdata_new)

for x, y, z, label in zip(xdata_new, ydata_new, zdata_new, itop_labels):
    ax.text(x, y, z, label)

ax.text2D(0.05, 0.95, "PANOPTIC WITH ITOP SKELETON MAPPING", transform=ax.transAxes)

ax.set_xlabel('x', rotation=0, fontsize=20, labelpad=20)
ax.set_ylabel('y', rotation=0, fontsize=20, labelpad=20)
ax.set_zlabel('z', rotation=0, fontsize=20, labelpad=20)
ax.set_xlim3d(-0.5,0.5)
ax.set_ylim3d(-2,2)
ax.set_zlim3d(0,2)
plt.show(block=False)
input('press <ENTER> to continue')
