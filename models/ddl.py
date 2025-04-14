import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import numpy as np
import os
import h5py


class DDL(BaseVAE):

    def __init__(self, 
                temporal_resolution: int,
                **kwargs) -> None:
        super(DDL, self).__init__()

        self.temporal_resolution = temporal_resolution

        # Define the activations
        self.relu = nn.ReLU()

        # Build Encoder (N, C, D, H, W)
        self.conv1  = nn.LazyConv3d(32, (3, 3, 3), padding=(1, 1, 1))
        if self.temporal_resolution == 10:
            self.pool1 = nn.MaxPool3d((1,2,2))
        elif self.temporal_resolution == 20:
            self.pool1 = nn.MaxPool3d(2)
        else:
            raise ValueError(f"Invalid temporal_resolution: {self.temporal_resolution}. Expected 10 or 20.")
        #
        self.conv2  = nn.LazyConv3d(64, (3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(2)
        #
        self.conv3  = nn.LazyConv3d(128, (3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d((1,2,2))
        #
        self.conv33 = nn.LazyConv3d(128, (3, 3, 3), padding=(1, 1, 1))
        #(N, C, D, H, W)

        self.lstm_layer = nn.LSTM(input_size=32768, hidden_size=2048, num_layers=1, bidirectional=False, batch_first=True)

        self.lstm_layer1 = nn.LSTM(input_size=2048, hidden_size=64, num_layers=1, bidirectional=False, batch_first=True)
        self.lstm_layer2 = nn.LSTM(input_size=2048, hidden_size=64, num_layers=1, bidirectional=False, batch_first=True)
        self.lstm_layer3 = nn.LSTM(input_size=2048, hidden_size=64, num_layers=1, bidirectional=False, batch_first=True)

        self.mean1 = nn.LazyLinear(2)
        self.mean2 = nn.LazyLinear(1)
        self.mean3 = nn.LazyLinear(2)


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Forward the input by passing through the encoder-decoder network
        :param input: (Tensor) Input tensor to encoder [N x C x D x H x W]
        :return: (Tensor) List of outputs
        """
        conv1 = self.relu(self.conv1(input)) # its shape is (batch_size, channels, depth, height, width)
        pool1 = self.pool1(conv1)
        #
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)
        #
        conv3 = self.relu(self.conv3(pool2))
        pool3 = self.pool3(conv3)
        #
        conv33 = self.relu(self.conv33(pool3))
    
        # Compute the number of features (channels*height*width)
        num_features = conv33.shape[1] * conv33.shape[3] * conv33.shape[4]
        # Reshape it for LSTM
        cnn_model = conv33.permute(0, 2, 1, 3, 4).contiguous().view(-1, conv33.shape[2], num_features) # (batch_size, depth, channels*height*width)
        # print(f"cnn_model shape: {cnn_model.shape}")

        lstm, _ = self.lstm_layer(cnn_model)
        # print(f"lstm output shape: {lstm.shape}")

        lstm1, _ = self.lstm_layer1(lstm)
        lstm2, _ = self.lstm_layer2(lstm)
        lstm3, _ = self.lstm_layer3(lstm)

        lstm1 = lstm1[:,-1,:]
        lstm2 = lstm2[:,-1,:]
        lstm3 = lstm3[:,-1,:]

        mean1 = self.mean1(lstm1)

        mean2 = self.mean2(lstm2)

        mean3 = self.mean3(lstm3)

        return [mean1, mean2, mean3]


    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        """
        cloud, params = kwargs['inputs']

        mean1 = args[0]
        mean2 = args[1]
        mean3 = args[2]
            
        xy_loss = F.mse_loss(mean1, params[:,:2])
        t_loss = F.mse_loss(mean2, params[:,2:3])
        uv_loss = F.mse_loss(mean3, params[:,3:])

        loss = xy_loss + t_loss + uv_loss
        
        return {'loss': loss, 'xy':xy_loss.detach(), 't':t_loss.detach(), 'uv':uv_loss.detach()}


