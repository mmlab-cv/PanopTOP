import numpy as np
from torch.utils.data import Dataset

from .DBDataset import DBDataset
from .PoseDataset import PoseDataset

V2V = "V2V"

class DatasetFactory():
    def __init__(self, train_dataset, validation_dataset, test_dataset, transform_train, transform_val, transform_test, network_setup = V2V):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.network_setup = network_setup

        assert(self.train_dataset in ["ITOP", "PANOPTIC", "BOTH"])
        assert(self.validation_dataset in ["ITOP", "PANOPTIC", "BOTH"])
        assert(self.test_dataset in ["ITOP", "PANOPTIC", "BOTH"])

        self.transform_train = transform_train
        self.transform_val = transform_val
        self.transform_test = transform_test

        if self.network_setup == V2V:
            x_type = "cloud"
            y_type = "joints"
        else:
            raise Exception

        train_dataset = DBDataset(self.train_dataset, "train", x_type, y_type)
        validation_dataset = DBDataset(self.validation_dataset, "validation", x_type, y_type)
        test_dataset = DBDataset(self.test_dataset, "test", x_type, y_type)

        self.train_X, self.train_Y, self.train_X_paths, self.train_Y_paths = train_dataset.get_dataset()
        self.val_X, self.val_Y, self.val_X_paths, self.val_Y_paths = validation_dataset.get_dataset()
        self.test_X, self.test_Y, self.test_X_paths, self.test_Y_paths = test_dataset.get_dataset()
       

    def get_train(self):
        if self.network_setup == V2V:
            return PoseDataset(self.train_X, self.train_Y, self.transform_train, self.train_X_paths, self.train_Y_paths, 5)


    def get_validation(self):
        if self.network_setup == "V2V":
            return PoseDataset(self.val_X, self.val_Y, self.transform_val, self.val_X_paths, self.val_Y_paths)


    def get_test(self):
        if self.network_setup == "V2V":
            return PoseDataset(self.test_X, self.test_Y, self.transform_test, self.test_X_paths, self.test_Y_paths)
