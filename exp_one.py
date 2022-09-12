import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.autoencoder import TAE, ClusterNet
from datasets.datasets import BeetleFlyDataset


def pretrain_tae(model, loader, n_epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    pbar = tqdm(range(n_epochs))

    for i in pbar:
        for x, y in loader:
            x = x.permute(0, 2, 1)
            l, x_prime = tae(x)

            x_prime = x_prime.unsqueeze(1)

            loss = loss_fn(x_prime, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.set_description(f"{round(loss.item(), 6)}")


def kl_loss_function(input, pred):
    out = input * torch.log((input) / (pred))
    return torch.mean(torch.sum(out, dim=1))


if __name__ == "__main__":
    dataset = BeetleFlyDataset(
        train_path="data/BeetleFly/BeetleFly_TRAIN",
        test_path="data/BeetleFly/BeetleFly_TEST",
    )
    loader = DataLoader(dataset=dataset, batch_size=40)

    tae = TAE(pooling=8, filter_1=50, filter_lstm=[50, 1], seq_len=512, n_hidden=64)

    pretrain_tae(tae, loader, n_epochs=10)

    cl = ClusterNet(tae=tae, n_hidden=64, n_clusters=2, similarity="EUC")

    for x, y in loader:
        cl.init_centroids(x.permute(0, 2, 1))

    optimizer = optim.SGD(cl.parameters(), lr=0.01)
    mse_loss_fn = nn.MSELoss()

    max_method = None
    max_auc = -1

    pbar = tqdm(range(10))

    for i in pbar:
        all_probas = []
        all_preds = []
        all_labels = []

        for x, y in loader:
            x = x.permute(0, 2, 1)
            l, x_prime, Q, P = cl(x)
            x_prime = x_prime.unsqueeze(1)

            loss_mse = mse_loss_fn(x_prime, x)
            loss_kl = kl_loss_function(P, Q)

            total_loss = loss_mse + loss_kl

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            all_preds.append(torch.max(Q, dim=1)[1])
            all_probas.append(Q[:, 1].detach())
            all_labels.append(y - 1)

        all_preds = torch.cat(all_preds).numpy()
        all_probas = torch.cat(all_probas).numpy()
        all_labels = torch.cat(all_labels).numpy()

        roc_probas = max(
            roc_auc_score(all_labels, all_probas),
            roc_auc_score(all_labels, 1 - all_probas),
        )
        # roc_preds = max(
        #     roc_auc_score(all_labels, all_preds),
        #     roc_auc_score(all_labels, 1 - all_preds),
        # )

        if roc_probas > max_auc:
            max_auc = roc_probas
            max_method = "Probability"

        # if roc_preds > max_auc:
        #     max_auc = roc_preds
        #     max_method = "Thresholded"

        pbar.set_description(f"{round(max_auc, 6)}")

    with open("results.csv", "a") as f:
        f.write(f"{max_auc},{max_method}\n")

