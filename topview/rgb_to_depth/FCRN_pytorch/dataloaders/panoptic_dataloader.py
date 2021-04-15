import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader


class PANOPTICDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(PANOPTICDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (400, 400)

    def train_transform(self, rgb, depth):
        return rgb, depth

    def val_transform(self, rgb, depth):
        return rgb, depth
