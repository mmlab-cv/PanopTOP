"""
=============================
Written by: Giulia Martinelli
=============================
==================
About this Script:
==================

This script file defines a function that takes as input a numpy array (19,3), that contains the joints
coordinates with Panoptic format, and return a numpy array (15,3) that contains the Panoptic joints converted in
Itop format.

"""
import math as m
import numpy as np
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

R = Ry(-m.pi/2)*Rx(m.pi)

def change_format(joints):
    panoptic_allign = np.empty(shape=(19, 3), dtype=float)
    for index in range(len(joints)):
        panoptic_allign[index] = joints[:,:3][index]*R
        
    xdata = panoptic_allign[:,0]/100
    ydata = -panoptic_allign[:,2]/100
    zdata = -panoptic_allign[:,1]/100+2.5
    panoptic_head = [(xdata[16]+xdata[18])/2,(ydata[16]+ydata[18])/2,(zdata[16]+zdata[18])/2]
    panoptic_torso = [(xdata[0]+xdata[2])/2,(ydata[0]+ydata[2])/2,(zdata[0]+zdata[2])/2]

    xdata_new = np.array([panoptic_head[0], xdata[0], xdata[9], xdata[3], xdata[10], xdata[4], xdata[11], xdata[5], panoptic_torso[0], xdata[12], xdata[6], xdata[13], xdata[7], xdata[14], xdata[8]])
    ydata_new = np.array([panoptic_head[1], ydata[0], ydata[9], ydata[3], ydata[10], ydata[4], ydata[11], ydata[5], panoptic_torso[1], ydata[12], ydata[6], ydata[13], ydata[7], ydata[14], ydata[8]])
    zdata_new = np.array([panoptic_head[2], zdata[0], zdata[9], zdata[3], zdata[10], zdata[4], zdata[11], zdata[5], panoptic_torso[2], zdata[12], zdata[6], zdata[13], zdata[7], zdata[14], zdata[8]])

    panoptic_converted = np.empty(shape=(15, 3), dtype=float)
    for index in range(len(panoptic_converted)):
        panoptic_converted[index,0] = xdata_new[index]
        panoptic_converted[index,1] = ydata_new[index]
        panoptic_converted[index,2] = zdata_new[index]

    return panoptic_converted

if __name__ == '__main__':
    P = gzip.GzipFile('/home/Datasets/PANOPTIC/joints/00_00000000.npy.gz', "r")
    panoptic = np.load(P)
    joints = change_format(panoptic)
    print(joints)
