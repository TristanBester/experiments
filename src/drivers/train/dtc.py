import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score


def train_dtc(model, decoder, lr, n_epochs, loader, device):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)

    max_auc = -1

    for i in range(n_epochs):
        all_preds = []
        all_labels = []

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            l, Q, P = model(x)
            x_prime = decoder(l)
            log_Q = torch.log(Q)
            log_P = torch.log(P)

            loss_mse = mse_loss_fn(x_prime, x)
            loss_kl = kl_loss_fn(log_Q, log_P)
            loss = loss_mse + loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_preds.append(Q)
            all_labels.append(y - 1)

        y_hat = torch.cat(all_preds).detach().cpu().numpy()
        y = torch.cat(all_labels).detach().cpu().numpy()

        y_hat = y_hat[:, 1]

        auc = max(roc_auc_score(y, y_hat), roc_auc_score(1 - y, y_hat))

        if auc > max_auc:
            max_auc = auc

    return max_auc
