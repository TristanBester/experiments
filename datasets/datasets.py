import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class UCRDataset(Dataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.device = device

        self._load_data()

    def _load_as_ndarray(self, path):
        with open(path, "r") as f:
            data = list(csv.reader(f, delimiter=","))
        return np.array(data)

    def _split_features_target(self, arr):
        Y = arr[:, 0]
        X = arr[:, 1:]
        return X.astype(float), Y.astype(float).flatten()

    def _load_data(self):
        arr_train = self._load_as_ndarray(self.train_path)
        arr_test = self._load_as_ndarray(self.test_path)

        X_train, y_train = self._split_features_target(arr_train)
        X_test, y_test = self._split_features_target(arr_test)

        self.X = torch.from_numpy(np.vstack((X_train, X_test)))
        self.Y = torch.from_numpy(np.concatenate((y_train, y_test)))

        self.X = self.X.float()
        self.Y = self.Y.float()

        if self.device:
            self.X = self.X.to(self.device)
            self.Y = self.y.to(self.device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index].unsqueeze(-1)
        y = self.Y[index]
        return x, y


class BeetleFlyDataset(UCRDataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__(train_path, test_path, device)


class BirdChickenDataset(UCRDataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__(train_path, test_path, device)


class ComputersDataset(UCRDataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__(train_path, test_path, device)
