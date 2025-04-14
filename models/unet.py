from re import S
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import numpy as np
import os

class Unet(BaseVAE):

    def __init__(self, 
                temporal_resolution: int,
                **kwargs) -> None:
        super(Unet, self).__init__()

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

        # Build Decoder
        self.up4_ =  nn.Upsample(scale_factor=(1,2,2))
        self.up4 =   nn.LazyConv3d(128, (3, 3, 3), padding=(1, 1, 1))
        self.conv4 = nn.LazyConv3d(64, (3, 3, 3), padding=(1, 1, 1))
        #
        self.up5_ =  nn.Upsample(scale_factor=2)
        self.up5 =   nn.LazyConv3d(64, (3, 3, 3), padding=(1, 1, 1))
        self.conv5 = nn.LazyConv3d(32, (3, 3, 3), padding=(1, 1, 1))
        #
        if self.temporal_resolution == 10:
            self.up6_ =  nn.Upsample(scale_factor=(1,2,2))
        elif self.temporal_resolution == 20:
            self.up6_ =  nn.Upsample(scale_factor=2)
        else:
            raise ValueError(f"Invalid temporal_resolution: {self.temporal_resolution}. Expected 10 or 20.")
        self.up6 =   nn.LazyConv3d(32, (3, 3, 3), padding=(1, 1, 1))
        self.conv6 = nn.LazyConv3d(16, (3, 3, 3), padding=(1, 1, 1))
        # Define the output layer of the network
        self.output_layer = nn.LazyConv3d(1, (1, 1, 1), padding=(0, 0, 0))


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
    
        up4_ = self.up4_(conv33)
        up4 = self.relu(self.up4(up4_))
        concat4 = torch.cat((conv3,up4), 1)
        conv4 = self.relu(self.conv4(concat4))
        #
        up5_ = self.up5_(conv4)
        up5 = self.relu(self.up5(up5_))
        concat5 = torch.cat((conv2,up5), 1)
        conv5 = self.relu(self.conv5(concat5))
        #
        up6_ = self.up6_(conv5)
        up6 = self.relu(self.up6(up6_))
        concat6 = torch.cat((conv1,up6), 1)
        conv6 = self.relu(self.conv6(concat6))
        #
        output = self.output_layer(conv6)

        return [output]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the loss function.
        """
        _, cloud = kwargs['inputs']
        recons = args[0]

        loss =F.mse_loss(recons, cloud)
            
        return {'loss': loss}





