import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom.autoencoder import DTC, Autoencoder
from datasets import BeetleFlyDataset


def euclidean_distance(x, y):
    return torch.sum((x - y) ** 2)


def correlation_based_similarity(x, y):
    t = torch.vstack((x.flatten(), y.flatten()),)
    p = torch.corrcoef(t)[0][1]
    return torch.sqrt(2 * (1 - p))


def _complexity_estimate(x):
    x_back_shift = x[:-1]
    x_forward_shift = x[1:]
    return torch.sqrt(torch.sum((x_forward_shift - x_back_shift) ** 2))


def _complexitity_factor(x, y):
    ce = torch.tensor([_complexity_estimate(x), _complexity_estimate(y)])
    return torch.max(ce) / (torch.min(ce) + 1e-8)


def complexity_invariant_similarity(x, y):
    ed = euclidean_distance(x, y)
    cf = _complexitity_factor(x, y)
    return ed * cf


def _calculate_similarity_matrix(X, metric):
    similarity_matrix = torch.zeros(size=(X.shape[0], X.shape[0]))

    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            similarity_matrix[i][j] = metric(X[i], X[j])
    return similarity_matrix


def calculate_centroids(latents, metric, k):
    similarity_matrix = _calculate_similarity_matrix(latents, metric)
    similarity_matrix = similarity_matrix.numpy().astype(float)

    clustering_assignments = AgglomerativeClustering(
        n_clusters=k, affinity="precomputed", linkage="complete",
    ).fit_predict(similarity_matrix)

    centroids = []
    for i in np.unique(clustering_assignments):
        centroid = (
            latents[clustering_assignments == i].mean(dim=0, dtype=float).unsqueeze(0)
        )
        centroids.append(centroid)
    centroids = torch.cat(centroids)
    return centroids


if __name__ == "__main__":
    dataset = BeetleFlyDataset(
        train_path="data/BeetleFly/BeetleFly_TRAIN",
        test_path="data/BeetleFly/BeetleFly_TEST",
    )
    loader = DataLoader(dataset=dataset, batch_size=40)

    ae = Autoencoder(
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

    optimizer = optim.SGD(ae.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    pbar = tqdm(range(10))

    for i in pbar:
        for x, y in loader:
            l, x_prime = ae(x)

            loss = loss_fn(x_prime, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.set_description(f"{round(loss.item(), 6)}")

    latents = []

    for i in range(len(dataset)):
        l, _ = ae(dataset[i][0].unsqueeze(0))
        latents.append(l.detach())
    latents = torch.cat(latents)

    centroids = calculate_centroids(latents, complexity_invariant_similarity, k=2)

    cl = DTC(ae.encoder, centroids=centroids, metric=complexity_invariant_similarity)

    optimizer = optim.SGD(cl.parameters(), lr=0.01)
    mse_loss_fn = nn.MSELoss()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)

    max_auc = -1

    pbar = tqdm(range(10))

    for i in pbar:
        all_probas = []
        all_labels = []

        for x, y in loader:
            l, P, Q = cl(x)
            x_prime = ae.decoder(l)

            loss_mse = mse_loss_fn(x_prime, x)
            loss_kl = kl_loss_fn(torch.log(Q), torch.log(P))

            loss = loss_mse + loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_probas.append(Q[:, 1].detach())
            all_labels.append(y - 1)

        all_probas = torch.cat(all_probas).numpy()
        all_labels = torch.cat(all_labels).numpy()

        roc_auc = max(
            roc_auc_score(all_labels, all_probas),
            roc_auc_score(all_labels, 1 - all_probas),
        )

        if roc_auc > max_auc:
            max_auc = roc_auc

        pbar.set_description(f"{max_auc}")

    with open("results2_cid.csv", "a") as f:
        f.write(f"{max_auc},N/A\n")

