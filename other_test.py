import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.datasets import BeetleFlyDataset
from src.drivers.train import pretrain_autoencoder
from src.methods.mine.cluster import calculate_centroids, get_latents
from src.methods.mine.models import DTC, Autoencoder

HPARAMS = {
    "AE": {
        "seq_len": 512,
        "input_dim": 1,
        "cnn_channels": 50,
        "cnn_kernel": 10,
        "cnn_stride": 1,
        "mp_kernel": 8,
        "mp_stride": 8,
        "lstm_hidden_dim": 50,
        "deconv_kernel": 10,
        "deconv_stride": 1,
    },
    "CL": {"n_clusters": 2, "similarity": "EUC"},
}

if __name__ == "__main__":
    dataset = BeetleFlyDataset(
        train_path="data/BeetleFly/BeetleFly_TRAIN",
        test_path="data/BeetleFly/BeetleFly_TEST",
    )
    loader = DataLoader(dataset, batch_size=40)
    device = torch.device("cpu")

    model = Autoencoder(**HPARAMS["AE"])

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    ae_loss = pretrain_autoencoder(
        model, optimizer, loss_fn, loader, device, n_epochs=10
    )

    print(ae_loss)

    latents = get_latents(model, loader, device)

    centroids = calculate_centroids(latents, "EUC", k=2)

    dtc = DTC(encoder=model.encoder, centroids=centroids, metric="EUC")

    print(dtc(dataset[0][0].unsqueeze(0)))
