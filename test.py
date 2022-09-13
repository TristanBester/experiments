import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.datasets import BeetleFlyDataset
from src.drivers.train import pretrain_autoencoder
from src.methods.other.cluster import calculate_centroids, get_latents
from src.methods.other.models import TAE, ClusterNet

if __name__ == "__main__":
    dataset = BeetleFlyDataset(
        train_path="data/BeetleFly/BeetleFly_TRAIN",
        test_path="data/BeetleFly/BeetleFly_TEST",
    )
    loader = DataLoader(dataset, batch_size=40)
    device = torch.device("cpu")

    model = TAE(seq_len=512, pooling=8, cnn_channels=50, lstm_hidden=50, n_hidden=64)

    latents = get_latents(model, loader, device)

    centroids = calculate_centroids(latents, metric="EUC", k=2)

    dtc = ClusterNet(
        tae=model, centroids=centroids, n_hidden=64, n_clusters=2, similarity="EUC"
    )

    print(dtc(dataset[0][0].unsqueeze(0)))

    # optim = optim.SGD(model.parameters(), lr=0.01)
    # loss = nn.MSELoss()

    # loss_one = pretrain_autoencoder(model, optim, loss, loader, device, n_epochs=10)

    # print(loss_one)

