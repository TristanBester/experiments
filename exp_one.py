import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.autoencoder import TAE
from custom.autoencoder import Autoencoder
from datasets import BeetleFlyDataset

if __name__ == "__main__":
    base_model = TAE(seq_len=512, pooling=8, n_hidden=64)
    custom_model = Autoencoder(
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

    dataset = BeetleFlyDataset(
        train_path="/Users/tristan/Documents/CS/Research/baselines/data/BeetleFly/BeetleFly_TRAIN",
        test_path="/Users/tristan/Documents/CS/Research/baselines/data/BeetleFly/BeetleFly_TEST",
    )
    loader = DataLoader(dataset, batch_size=40)

    base_opt = optim.SGD(base_model.parameters(), lr=0.01)
    cust_opt = optim.SGD(custom_model.parameters(), lr=0.01)

    base_loss = nn.MSELoss()
    cust_loss = nn.MSELoss()

    min_loss_base = float("inf")
    min_loss_cust = float("inf")

    for i in range(100):
        base_loss_ave = 0
        cust_loss_ave = 0

        for x, y in loader:
            l_base, x_prime_base = base_model(x.permute(0, 2, 1))
            l_cust, x_prime_cust = custom_model(x)

            loss_base = base_loss(x_prime_base.unsqueeze(-1), x)
            loss_cust = cust_loss(x_prime_cust, x)

            base_opt.zero_grad()
            loss_base.backward()
            base_opt.step()

            cust_opt.zero_grad()
            loss_cust.backward()
            cust_opt.step()

            base_loss_ave += loss_base.item()
            cust_loss_ave += loss_cust.item()

        if base_loss_ave < min_loss_base:
            min_loss_base = base_loss_ave
        if cust_loss_ave < min_loss_cust:
            min_loss_cust = cust_loss_ave

        # print(base_loss_ave, cust_loss_ave)

    with open("results.txt", "a") as f:
        f.write(f"b: {min_loss_base}\tc: {min_loss_cust}\n")

    print("Complete.")
