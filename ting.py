import torch
from torch.utils.data import DataLoader

from src.datasets.datasets import BeetleFlyDataset
from src.methods.mine import init_DTC as init_my_DTC
from src.methods.other import init_DTC as init_other_DTC

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

    my_loss, my_decoder, my_dtc = init_my_DTC(
        ae_pretrain_lr=0.01,
        ae_pretrain_epochs=10,
        loader=loader,
        device=device,
        hparams=HPARAMS,
    )
    other_loss, other_decoder, other_dtc = init_other_DTC(
        ae_pretrain_lr=0.01,
        ae_pretrain_epochs=10,
        loader=loader,
        device=device,
        hparams=HPARAMS,
    )

    print(my_loss, other_loss)
    print(my_dtc(dataset[0][0].unsqueeze(0)))
    print(other_dtc(dataset[0][0].unsqueeze(0)))

