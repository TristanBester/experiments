import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering

from ..utils import compute_similarity


def get_latents(model, loader, device):
    latents = []

    for x, _ in loader:
        x = x.to(device)
        l, _ = model(x)
        latents.append(l)
    return torch.cat(latents).detach().cpu()


def calculate_centroids(latents, metric, k):
    similarity_matrix = compute_similarity(latents, latents, similarity=metric)

    assignements = AgglomerativeClustering(
        n_clusters=k, linkage="complete", affinity="precomputed"
    ).fit_predict(similarity_matrix)

    centroids_ = torch.zeros((k, latents.shape[1]))

    for cluster_ in range(k):
        index_cluster = [k for k, index in enumerate(assignements) if index == cluster_]
        centroids_[cluster_] = torch.mean(latents[index_cluster], dim=0)
    return centroids_

