import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        cnn_channels,
        cnn_kernel,
        cnn_stride,
        mp_kernel,
        mp_stride,
        lstm_hidden_dim,
    ) -> None:
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim

        self.cnn = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_channels,
            kernel_size=cnn_kernel,
            stride=cnn_stride,
        )
        self.max_pool = nn.MaxPool1d(kernel_size=mp_kernel, stride=mp_stride,)

        self.lstm_layer_one = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm_layer_two = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=1,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.leaky_relu(self.cnn(x))
        x = self.max_pool(x)

        x = x.permute(0, 2, 1)
        x, (_, _) = self.lstm_layer_one(x)
        x = x[:, :, : self.lstm_hidden_dim] + x[:, :, : self.lstm_hidden_dim]

        x, (_, _) = self.lstm_layer_two(x)
        x = (x[:, :, 0] + x[:, :, 1]).unsqueeze(-1)
        x = x.permute(0, 2, 1)
        return x


class Decoder(nn.Module):
    def __init__(self, output_size, deconv_kernel, deconv_stride) -> None:
        super().__init__()
        self.output_size = output_size
        self.kernel = deconv_kernel
        self.stride = deconv_stride

        self._calcuate_required_kernel()

        upsample = int((self.output_size + self.stride - self.kernel) / self.stride)

        self.upsample = nn.Upsample(size=(upsample,))
        self.deconv = nn.ConvTranspose1d(
            1, 1, kernel_size=self.kernel, stride=self.stride
        )

    def _calcuate_required_kernel(self):
        while True:
            l_up = (self.output_size + self.stride - self.kernel) / self.stride

            if l_up % 1 == 0:
                break
            else:
                self.kernel -= 1

    def forward(self, x):
        x = self.upsample(x)
        x = F.leaky_relu(self.deconv(x))
        return x.permute(0, 2, 1)


class Autoencoder(nn.Module):
    def __init__(
        self,
        seq_len,
        input_dim,
        cnn_channels,
        cnn_kernel,
        cnn_stride,
        mp_kernel,
        mp_stride,
        lstm_hidden_dim,
        deconv_kernel,
        deconv_stride,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            cnn_channels=cnn_channels,
            cnn_kernel=cnn_kernel,
            cnn_stride=cnn_stride,
            mp_kernel=mp_kernel,
            mp_stride=mp_stride,
            lstm_hidden_dim=lstm_hidden_dim,
        )
        self.decoder = Decoder(
            output_size=seq_len,
            deconv_kernel=deconv_kernel,
            deconv_stride=deconv_stride,
        )

    def forward(self, x):
        l = self.encoder(x)
        r = self.decoder(l)
        return l.permute(0, 2, 1), r


class DTC(nn.Module):
    def __init__(self, encoder, centroids, metric) -> None:
        super().__init__()
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
