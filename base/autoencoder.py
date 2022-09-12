import gc

import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering


class TAE_encoder(nn.Module):
    """
    Class for temporal autoencoder encoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    pooling : pooling number for maxpooling.
    """

    def __init__(self, filter_1, filter_lstm, pooling):
        super().__init__()

        self.hidden_lstm_1 = filter_lstm[0]
        self.hidden_lstm_2 = filter_lstm[1]
        self.pooling = pooling
        self.n_hidden = None
        ## CNN PART
        ### output shape (batch_size, 50 , n_hidden = 64)
        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=filter_1,
                kernel_size=10,
                stride=1,
                padding=5,
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.pooling),
        )

        ## LSTM PART
        ### output shape (batch_size , n_hidden = 64 , 50)
        self.lstm_1 = nn.LSTM(
            input_size=50,
            hidden_size=self.hidden_lstm_1,
            batch_first=True,
            bidirectional=True,
        )

        ### output shape (batch_size , n_hidden = 64 , 1)
        self.lstm_2 = nn.LSTM(
            input_size=50,
            hidden_size=self.hidden_lstm_2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):

        ## encoder
        out_cnn = self.conv_layer(x)
        out_cnn = out_cnn.permute((0, 2, 1))
        out_lstm1, _ = self.lstm_1(out_cnn)
        out_lstm1 = torch.sum(
            out_lstm1.view(
                out_lstm1.shape[0], out_lstm1.shape[1], 2, self.hidden_lstm_1
            ),
            dim=2,
        )
        features, _ = self.lstm_2(out_lstm1)
        features = torch.sum(
            features.view(features.shape[0], features.shape[1], 2, self.hidden_lstm_2),
            dim=2,
        )  ## (batch_size , n_hidden ,1)
        if self.n_hidden == None:
            self.n_hidden = features.shape[1]
        return features


class TAE_decoder(nn.Module):
    """
    Class for temporal autoencoder decoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, n_hidden=64, pooling=8):
        super().__init__()

        self.pooling = pooling
        self.n_hidden = n_hidden

        # upsample
        self.up_layer = nn.Upsample(size=pooling)
        self.deconv_layer = nn.ConvTranspose1d(
            in_channels=self.n_hidden,
            out_channels=self.n_hidden,
            kernel_size=10,
            stride=1,
            padding=self.pooling // 2,
        )

    def forward(self, features):

        upsampled = self.up_layer(features)  ##(batch_size  , n_hidden , pooling)
        out_deconv = self.deconv_layer(upsampled)[:, :, : self.pooling].contiguous()
        out_deconv = out_deconv.view(out_deconv.shape[0], -1)
        return out_deconv


class TAE(nn.Module):
    """
    Class for temporal autoencoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, seq_len, pooling, n_hidden, filter_1=50, filter_lstm=[50, 1]):
        super().__init__()

        self.pooling = pooling
        self.filter_1 = filter_1
        self.filter_lstm = filter_lstm

        self.tae_encoder = TAE_encoder(
            filter_1=self.filter_1, filter_lstm=self.filter_lstm, pooling=self.pooling,
        )
        n_hidden = self.get_hidden(seq_len, "cpu")

        self.tae_decoder = TAE_decoder(n_hidden=n_hidden, pooling=self.pooling)

    def get_hidden(self, serie_size, device):
        a = torch.randn((1, 1, serie_size)).to(device)
        test_model = TAE_encoder(
            filter_1=self.filter_1, filter_lstm=self.filter_lstm, pooling=self.pooling,
        ).to(device)
        with torch.no_grad():
            _ = test_model(a)
        n_hid = test_model.n_hidden
        del test_model, a
        gc.collect()
        torch.cuda.empty_cache()
        return n_hid

    def forward(self, x):

        features = self.tae_encoder(x)
        out_deconv = self.tae_decoder(features)
        return features.squeeze(2), out_deconv


class ClusterNet(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self, tae, n_hidden, n_clusters, similarity):
        super().__init__()

        ## init with the pretrained autoencoder model
        self.tae = tae

        ## clustering model
        self.alpha_ = 1
        self.centr_size = n_hidden
        self.n_clusters = n_clusters
        self.device = "cpu"
        self.similarity = similarity

    def init_centroids(self, x):
        """
        This function initializes centroids with agglomerative clustering
        + complete linkage.
        """
        z, _ = self.tae(x)
        z_np = z.detach().cpu()
        assignements = AgglomerativeClustering(
            n_clusters=2, linkage="complete", affinity="precomputed"
        ).fit_predict(compute_similarity(z_np, z_np, similarity=self.similarity))

        centroids_ = torch.zeros((self.n_clusters, self.centr_size), device=self.device)

        for cluster_ in range(self.n_clusters):
            index_cluster = [
                k for k, index in enumerate(assignements) if index == cluster_
            ]
            centroids_[cluster_] = torch.mean(z.detach()[index_cluster], dim=0)

        self.centroids = nn.Parameter(centroids_)

    def forward(self, x):

        z, x_reconstr = self.tae(x)
        z_np = z.detach().cpu()

        similarity = compute_similarity(z, self.centroids, similarity=self.similarity)

        ## Q (batch_size , n_clusters)
        Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)
        sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)
        Q = Q / sum_columns_Q

        ## P : ground truth distribution
        P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1, -1)
        sum_columns_P = torch.sum(P, dim=1).view(-1, 1)
        P = P / sum_columns_P
        return z, x_reconstr, Q, P


def compute_CE(x):
    """
    x shape : (n , n_hidden)
    return : output : (n , 1)
    """
    return torch.sqrt(torch.sum(torch.square(x[:, 1:] - x[:, :-1]), dim=1))


def compute_similarity(z, centroids, similarity="EUC"):
    """
    Function that compute distance between a latent vector z and the clusters centroids.

    similarity : can be in [CID,EUC,COR] :  euc for euclidean,  cor for correlation and CID
                 for Complexity Invariant Similarity.
    z shape : (batch_size, n_hidden)
    centroids shape : (n_clusters, n_hidden)
    output : (batch_size , n_clusters)
    """
    n_clusters, n_hidden = centroids.shape[0], centroids.shape[1]
    bs = z.shape[0]

    if similarity == "CID":
        CE_z = compute_CE(z).unsqueeze(1)  # shape (batch_size , 1)
        CE_cen = compute_CE(centroids).unsqueeze(0)  ## shape (1 , n_clusters )
        z = z.unsqueeze(0).expand((n_clusters, bs, n_hidden))
        mse = torch.sqrt(torch.sum((z - centroids.unsqueeze(1)) ** 2, dim=2))
        CE_z = CE_z.expand((bs, n_clusters))  # (bs , n_clusters)
        CE_cen = CE_cen.expand((bs, n_clusters))  # (bs , n_clusters)
        CF = torch.max(CE_z, CE_cen) / torch.min(CE_z, CE_cen)
        return torch.transpose(mse, 0, 1) * CF

    elif similarity == "EUC":
        z = z.expand((n_clusters, bs, n_hidden))
        mse = torch.sqrt(torch.sum((z - centroids.unsqueeze(1)) ** 2, dim=2))
        return torch.transpose(mse, 0, 1)

    elif similarity == "COR":
        std_z = (
            torch.std(z, dim=1).unsqueeze(1).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        mean_z = (
            torch.mean(z, dim=1).unsqueeze(1).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        std_cen = (
            torch.std(centroids, dim=1).unsqueeze(0).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        mean_cen = (
            torch.mean(centroids, dim=1).unsqueeze(0).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        ## covariance
        z_expand = z.unsqueeze(1).expand((bs, n_clusters, n_hidden))
        cen_expand = centroids.unsqueeze(0).expand((bs, n_clusters, n_hidden))
        prod_expec = torch.mean(z_expand * cen_expand, dim=2)  ## (bs , n_clusters)
        pearson_corr = (prod_expec - mean_z * mean_cen) / (std_z * std_cen)
        return torch.sqrt(2 * (1 - pearson_corr))
