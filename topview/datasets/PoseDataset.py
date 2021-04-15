import numpy as np
from torch.utils.data import Dataset

from .common import get_center_of_point_cloud



class PoseDataset(Dataset):
    def __init__(self, points, joints, transform, cloud_path, joints_path, augmentation_multiplier = None):
        self.points = points
        self.joints = joints
        self.cloud_path = cloud_path
        self.joints_path = joints_path
        self.transform = transform
        self.augmentation_multiplier = augmentation_multiplier

    def __len__(self):
        if self.augmentation_multiplier is not None:
            return self.augmentation_multiplier * len(self.points)
        else:
            return len(self.points)

    def __getitem__(self, index):
        if self.augmentation_multiplier:
            index = index // self.augmentation_multiplier
            
        sample = {
            'points': self.points[index],
            'joints': self.joints[index],
            'refpoint': get_center_of_point_cloud(self.joints[index]),
            'cloud_path': self.cloud_path[index],
            'joint_path': self.joints_path[index]
        }

        if self.transform: 
            sample = self.transform(sample)
            
        return sample


    def get_points(self):
        return self.points


    def get_joints(self):
        return self.joints