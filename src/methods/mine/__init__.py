import torch.nn as nn
import torch.optim as optim

from ...drivers.train import pretrain_autoencoder
from .cluster import calculate_centroids, get_latents
from .models import DTC, Autoencoder


def init_DTC(ae_pretrain_lr, ae_pretrain_epochs, loader, device, hparams):
    model = Autoencoder(**hparams["AE"])

    optimizer = optim.SGD(model.parameters(), lr=ae_pretrain_lr)
    loss_fn = nn.MSELoss()

    ae_loss = pretrain_autoencoder(
        model, optimizer, loss_fn, loader, device, n_epochs=ae_pretrain_epochs
    )

    latents = get_latents(model, loader, device)
    centroids = calculate_centroids(
        latents, metric=hparams["CL"]["metric"], k=hparams["CL"]["n_clusters"]
    )

    dtc = DTC(
        encoder=model.encoder, centroids=centroids, metric=hparams["CL"]["metric"],
    )
    return ae_loss, model.decoder, dtc
