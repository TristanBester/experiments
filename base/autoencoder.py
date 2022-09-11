import gc

import torch
import torch.nn as nn


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
