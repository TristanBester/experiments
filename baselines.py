import torch
from bson.objectid import ObjectId
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import BeetleFlyDataset, BirdChickenDataset, ComputersDataset
from src.drivers.train import train_dtc
from src.logging import create_experiment, log_result
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
    "pretrain": {"lr": 0.01, "n_epochs": 10},
    "train": {"lr": 0.01, "n_epochs": 10},
    "loader": {"batch_size": 40},
}


def create_model_baselines(model_name, model_factory, loader, device, exp_id):
    for _ in tqdm(range(500)):
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
            n_epochs=HPARAMS["train"]["n_epochs"],
            loader=loader,
            device=device,
        )

        log_result(
            model_name=model_name,
            result={
                "exp_id": ObjectId(exp_id),
                "dataset": "beetle_fly",
                "max_auc": max_auc,
                "aucs": aucs,
                "ae_loss": ae_loss,
            },
        )


if __name__ == "__main__":
    dataset = BeetleFlyDataset(
        train_path="data/BeetleFly/BeetleFly_TRAIN",
        test_path="data/BeetleFly/BeetleFly_TEST",
    )
    loader = DataLoader(dataset, batch_size=HPARAMS["loader"]["batch_size"])
    device = torch.device("cpu")

    exp_id = create_experiment("beetle_fly", HPARAMS)

    print("Creating baselines for DTC implementation...")
    create_model_baselines("my_dtc", init_my_dtc, loader, device, exp_id)

    print("Creating baselines for Other implementation...")
    create_model_baselines("other_dtc", init_other_dtc, loader, device, exp_id)
