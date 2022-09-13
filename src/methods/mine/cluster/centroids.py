import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from ..metrics import (
    complexity_invariant_similarity,
    correlation_based_similarity,
    euclidean_distance,
)
from .utils import calculate_similarity_matrix


def calculate_centroids(latents, metric, k):
    if metric == "EUC":
        metric = euclidean_distance
    elif metric == "CID":
        metric = complexity_invariant_similarity
    elif metric == "COR":
        metric = correlation_based_similarity
    else:
        raise ValueError("Invalid metric")

    similarity_matrix = calculate_similarity_matrix(latents, metric)
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
