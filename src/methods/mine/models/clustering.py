import torch
import torch.nn as nn

from ..metrics import (
    complexity_invariant_similarity,
    correlation_based_similarity,
    euclidean_distance,
)


class DTC(nn.Module):
    def __init__(self, encoder, centroids, metric) -> None:
        super().__init__()

        if metric == "EUC":
            metric = euclidean_distance
        elif metric == "CID":
            metric = complexity_invariant_similarity
        elif metric == "COR":
            metric = correlation_based_similarity
        else:
            raise ValueError("Invalid metric")

        self.encoder = encoder
        self.centroids = nn.Parameter(centroids)
        self.metric = metric

    def dist_to_centroids(self, x):
        dists = torch.zeros(x.shape[0], self.centroids.shape[0])

        for i in range(x.shape[0]):
            for j in range(self.centroids.shape[0]):
                dists[i][j] = self.metric(x[i], self.centroids[j])
        return dists

    def students_t_distribution_kernel(self, x, alpha=1):
        num = torch.pow((1 + x / alpha), -(alpha + 1) / 2)
        denom = num.sum(dim=1).reshape(-1, 1).repeat(1, self.centroids.shape[0])
        return num / denom

    def target_distribution(self, Q):
        F = Q.sum(dim=0)
        num = (Q ** 2) / F
        denom = num.sum(dim=1).reshape(-1, 1).repeat(1, Q.shape[-1])
        return num / denom

    def forward(self, x):
        l = self.encoder(x)
        D = self.dist_to_centroids(l)
        Q = self.students_t_distribution_kernel(D)
        P = self.target_distribution(Q)
        return l, Q, P
