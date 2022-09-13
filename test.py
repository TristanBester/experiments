import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.datasets import BeetleFlyDataset
from src.drivers.train import pretrain_autoencoder
from src.methods.mine.models import Autoencoder
from src.methods.other import init_DTC
from src.methods.other.cluster import calculate_centroids, get_latents
from src.methods.other.models import TAE, ClusterNet

Autoencoder(
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


# def init_DTC(ae_pretrain_lr, ae_pretrain_epochs, loader, device):
#     model = TAE(**HPARAMS["AE"])

#     optimizer = optim.SGD(model.parameters(), lr=ae_pretrain_lr)
#     loss_fn = nn.MSELoss()

#     ae_loss = pretrain_autoencoder(
#         model, optimizer, loss_fn, loader, device, n_epochs=ae_pretrain_epochs
#     )

#     latents = get_latents(model, loader, device)
#     centroids = calculate_centroids(
#         latents, metric=HPARAMS["CL"]["similarity"], k=HPARAMS["CL"]["n_clusters"]
#     )

#     dtc = ClusterNet(
#         encoder=model.tae_encoder,
#         centroids=centroids,
#         similarity=HPARAMS["CL"]["similarity"],
#     )
#     return ae_loss, dtc


if __name__ == "__main__":
    dataset = BeetleFlyDataset(
        train_path="data/BeetleFly/BeetleFly_TRAIN",
        test_path="data/BeetleFly/BeetleFly_TEST",
    )
    loader = DataLoader(dataset, batch_size=40)
    device = torch.device("cpu")

    loss, dtc = init_DTC(
        ae_pretrain_lr=0.1,
        ae_pretrain_epochs=10,
        loader=loader,
        device=device,
        hparams=HPARAMS,
    )

    # loss, dtc = init_DTC(
    #     ae_pretrain_lr=0.01, ae_pretrain_epochs=10, loader=loader, device=device
    # )

    print(loss)

    # model = TAE(seq_len=512, pooling=8, cnn_channels=50, lstm_hidden=50, n_hidden=64)

    # model = TAE(**HPARAMS["AE"])
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # loss_fn = nn.MSELoss()
    # print(pretrain_autoencoder(model, optimizer, loss_fn, loader, device, n_epochs=10))

    # latents = get_latents(model, loader, device)
    # centroids = calculate_centroids(
    #     latents, metric=HPARAMS["CL"]["similarity"], k=HPARAMS["CL"]["n_clusters"]
    # )

    # dtc = ClusterNet(
    #     encoder=model.tae_encoder,
    #     centroids=centroids,
    #     similarity=HPARAMS["CL"]["similarity"],
    # )

    # # print(dtc§)

    # # dtc = ClusterNet(
    # #     tae=model, centroids=centroids, n_hidden=64, n_clusters=2, similarity="EUC"
    # # )
    # #
    # print(dtc(dataset[0][0].unsqueeze(0)))

    # optim = optim.SGD(model.parameters(), lr=0.01)
    # loss = nn.MSELoss()

    # loss_one = pretrain_autoencoder(model, optim, loss, loader, device, n_epochs=10)

    # print(loss_one)

