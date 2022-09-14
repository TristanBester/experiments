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

        self.name = "BeetleFly"
        self.seq_len = 512
        self.pooling = 8


class BirdChickenDataset(UCRDataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__(train_path, test_path, device)

        self.name = "BirdChicken"
        self.seq_len = 512
        self.pooling = 8


class ComputersDataset(UCRDataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__(train_path, test_path, device)

        self.name = "Computers"
        self.seq_len = 720
        self.pooling = 10


class EarthquakesDataset(UCRDataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__(train_path, test_path, device)

        self.name = "Earthquakes"
        self.seq_len = 512
        self.pooling = 8


class MoteStrainDataset(UCRDataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__(train_path, test_path, device)

        self.name = "MoteStrain"
        self.seq_len = 84
        self.pooling = 4


class PhalangesOutlinesCorrectDataset(UCRDataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__(train_path, test_path, device)

        self.name = "PhalangesOutlinesCorrect"
        self.seq_len = 80
        self.pooling = 4


class ProximalPhalanxOutlineCorrectDataset(UCRDataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__(train_path, test_path, device)

        self.name = "ProximalPhalanxOutlineCorrect"
        self.seq_len = 80
        self.pooling = 4


class ShapeletSimDataset(UCRDataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__(train_path, test_path, device)

        self.name = "ShapeletSim"
        self.seq_len = 500
        self.pooling = 10


class SonyAIBORobotSurfaceDataset(UCRDataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__(train_path, test_path, device)

        self.name = "SonyAIBORobotSurface"
        self.seq_len = 70
        self.pooling = 5


class SonyAIBORobotSurfaceIIDataset(UCRDataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__(train_path, test_path, device)

        self.name = "SonyAIBORobotSurfaceII"
        self.seq_len = 65
        self.pooling = 5


class ItalyPowerDemandDataset(UCRDataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__(train_path, test_path, device)

        self.name = "ItalyPowerDemand"
        self.seq_len = 24
        self.pooling = 4


class WormsTwoClassDataset(UCRDataset):
    def __init__(self, train_path, test_path, device=None):
        super().__init__(train_path, test_path, device)

        self.name = "WormsTwoClass"
        self.seq_len = 900
        self.pooling = 10

