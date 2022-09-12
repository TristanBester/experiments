import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.datasets import BeetleFlyDataset
from src.drivers.train import pretrain_autoencoder
from src.methods.mine.models import Autoencoder
from src.methods.other.models import TAE

if __name__ == "__main__":
    dataset = BeetleFlyDataset(
        train_path="data/BeetleFly/BeetleFly_TRAIN",
        test_path="data/BeetleFly/BeetleFly_TEST",
    )
    loader = DataLoader(dataset, batch_size=40)
    device = torch.device("cpu")

    other_model = TAE(
        seq_len=512, pooling=8, cnn_channels=50, lstm_hidden=50, n_hidden=64
    )
    my_model = Autoencoder(
        seq_len=512,
        input_dim=1,
        cnn_channels=50,
        cnn_kernel=10,
        cnn_stride=1,
        mp_kernel=8,
        mp_stride=8,
        lstm_hidden_dim=50,
        deconv_kernel=10,
        deconv_stride=1,
    )

    other_optim = optim.SGD(other_model.parameters(), lr=0.01)
    my_optim = optim.SGD(my_model.parameters(), lr=0.01)

    other_loss = nn.MSELoss()
    my_loss = nn.MSELoss()

    loss_one = pretrain_autoencoder(
        other_model, other_optim, other_loss, loader, device, n_epochs=10
    )
    loss_two = pretrain_autoencoder(
        my_model, my_optim, my_loss, loader, device, n_epochs=10
    )

    print(loss_one, loss_two)

