import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader


class KITTIDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(KITTIDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (228, 912)

    def train_transform(self, rgb, depth):
        return rgb, depth

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Crop(130, 10, 240, 1200),
            transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = np.asfarray(depth_np, dtype='float32')
        depth_np = transform(depth_np)

        return rgb_np, depth_np
