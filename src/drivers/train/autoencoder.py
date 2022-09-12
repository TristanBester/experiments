def pretrain_autoencoder(model, optimizer, criterion, loader, device, n_epochs):
    model.train()

    for _ in range(n_epochs):
        for x, _ in loader:
            x = x.to(device)
            _, x_prime = model(x)

            loss = criterion(x_prime, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss.item()
