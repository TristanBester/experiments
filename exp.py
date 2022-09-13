import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.datasets import BeetleFlyDataset
from src.drivers.train import train_dtc
from src.methods.mine import init_DTC as init_my_dtc
from src.methods.other import init_DTC as init_other_dtc

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
    "CL": {"n_clusters": 2, "metric": "EUC"},
}


if __name__ == "__main__":
    dataset = BeetleFlyDataset(
        train_path="data/BeetleFly/BeetleFly_TRAIN",
        test_path="data/BeetleFly/BeetleFly_TEST",
    )
    loader = DataLoader(dataset, batch_size=40)
    device = torch.device("cpu")

    my_ae_loss, my_decoder, my_dtc = init_my_dtc(
        ae_pretrain_lr=0.01,
        ae_pretrain_epochs=10,
        loader=loader,
        device=device,
        hparams=HPARAMS,
    )
    other_ae_loss, other_decoder, other_dtc = init_other_dtc(
        ae_pretrain_lr=0.01,
        ae_pretrain_epochs=10,
        loader=loader,
        device=device,
        hparams=HPARAMS,
    )

    my_auc = train_dtc(
        my_dtc, my_decoder, lr=0.01, n_epochs=10, loader=loader, device=device
    )
    other_auc = train_dtc(
        other_dtc, other_decoder, lr=0.01, n_epochs=10, loader=loader, device=device
    )

    print(my_auc, other_auc)

