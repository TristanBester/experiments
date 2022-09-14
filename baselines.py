import argparse

import torch
from bson.objectid import ObjectId
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import BeetleFlyDataset, BirdChickenDataset, ComputersDataset
from src.drivers.data import init_data
from src.drivers.train import train_dtc
from src.logging import create_experiment, log_result
from src.methods.mine import init_DTC as init_my_dtc
from src.methods.other import init_DTC as init_other_dtc

HPARAMS = {
    "AE": {
        "input_dim": 1,
        "cnn_channels": 50,
        "cnn_kernel": 10,
        "cnn_stride": 1,
        "lstm_hidden_dim": 50,
        "deconv_kernel": 10,
        "deconv_stride": 1,
    },
    "CL": {"n_clusters": 2, "metric": "EUC"},
    "pretrain": {"lr": 0.01, "n_epochs": 10},
    "train": {"lr": 0.01, "n_epochs": 10},
    "loader": {"batch_size": 40},
}


def create_model_baseline(
    model_name, model_factory, dataset, loader, device, exp_id=None, n_repeats=1
):
    HPARAMS["AE"]["seq_len"] = dataset.seq_len
    HPARAMS["AE"]["mp_kernel"] = dataset.pooling
    HPARAMS["AE"]["mp_stride"] = dataset.pooling

    print(dataset.name)

    for _ in tqdm(range(n_repeats)):
        ae_loss, decoder, dtc = model_factory(
            ae_pretrain_lr=HPARAMS["pretrain"]["lr"],
            ae_pretrain_epochs=HPARAMS["pretrain"]["n_epochs"],
            loader=loader,
            device=device,
            hparams=HPARAMS,
        )

        max_auc, aucs = train_dtc(
            model=dtc,
            decoder=decoder,
            lr=HPARAMS["train"]["lr"],
            loader=loader,
            device=device,
        )

        if exp_id is not None:
            log_result(
                model_name=model_name,
                result={
                    "exp_id": ObjectId(exp_id),
                    "dataset": dataset.name,
                    "max_auc": max_auc,
                    "aucs": aucs,
                    "ae_loss": ae_loss,
                },
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_repeats", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    data = init_data(
        base_path="./data", batch_size=HPARAMS["loader"]["batch_size"], device=device
    )

    exp_id = create_experiment(exp_name="Baseline", hparams=HPARAMS)

    for dataset, loader in data:
        create_model_baseline(
            model_name="MyDTC",
            model_factory=init_my_dtc,
            dataset=dataset,
            loader=loader,
            device=device,
            exp_id=exp_id,
            n_repeats=args.n_repeats,
        )

