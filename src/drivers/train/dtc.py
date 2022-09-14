import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score


def train_dtc(model, decoder, lr, loader, device, max_epochs=100):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)

    max_auc = -1
    aucs = []
    last_assignments = None

    for _ in range(max_epochs):
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

        # Test for convergence
        assignments = torch.argmax(Q, dim=1)

        if last_assignments is not None:
            delta = torch.abs(assignments - last_assignments)

            if torch.sum(delta) < 0.001 * len(delta.flatten()):
                break

        last_assignments = assignments

        y_hat = y_hat[:, 1]

        try:
            auc = max(roc_auc_score(y, y_hat), roc_auc_score(1 - y, y_hat))
        except:
            auc = 0

        if auc > max_auc:
            max_auc = auc

        aucs.append(auc)

    return max_auc, aucs
