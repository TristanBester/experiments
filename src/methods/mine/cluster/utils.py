import torch


def calculate_similarity_matrix(X, metric):
    similarity_matrix = torch.zeros(size=(X.shape[0], X.shape[0]))

    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            similarity_matrix[i][j] = metric(X[i], X[j])
    return similarity_matrix


def get_latents(model, loader, device):
    latents = []

    for x, _ in loader:
        x = x.to(device)
        l, _ = model(x)
        latents.append(l)
    return torch.cat(latents).detach().cpu()
