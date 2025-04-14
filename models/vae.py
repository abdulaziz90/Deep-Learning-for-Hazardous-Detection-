import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import numpy as np
import os
import h5py

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.attention_score_layer = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x size: (batch_size, channels, depth, height, width)
        attention_scores = self.attention_score_layer(x)
        # attention_scores size: (batch_size, 1, depth, height, width)

        # Reshape to combine height and width
        attention_scores_reshaped = attention_scores.view(x.size(0), x.size(2), -1)
        # attention_scores_reshaped size: (batch_size, depth, height * width)

        # Apply softmax across the spatial dimensions (height and width)
        attention_weights = nn.functional.softmax(attention_scores_reshaped, dim=-1)
        # attention_weights size: (batch_size, depth, height * width)

        # Reshape back to original spatial dimensions
        attention_weights = attention_weights.view_as(attention_scores)
        # attention_weights size: (batch_size, 1, depth, height, width)

        attended_features = attention_weights * x
        # attended_features size: (batch_size, channels, depth, height, width)

        return attended_features

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention_layer = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs):
        # lstm_outputs shape: (batch_size, sequence_length, hidden_size)
        attention_scores = self.attention_layer(lstm_outputs) # shape: (batch_size, sequence_length, 1)
        attention_scores = attention_scores.squeeze(-1) # shape: (batch_size, sequence_length)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1) # shape: (batch_size, sequence_length)
        attention_weights = attention_weights.unsqueeze(1) # shape: (batch_size, 1, sequence_length)
        attended_features = torch.bmm(attention_weights, lstm_outputs) # shape: (batch_size, 1, hidden_size)
        attended_features = attended_features.squeeze(1) # shape: (batch_size, hidden_size)
        return attended_features


class VAE(BaseVAE):

    def __init__(self,
                data_path: str,
                temporal_resolution: int,
                latent_dim2: int,
                arch_option: str = 'NONE',
                lstm_option: str = 'NONE',
                cloud_weight: float = 1,
                kl_weight1: float = 1,
                kl_weight2: float = 1,
                param_weight: float = 1,
                **kwargs) -> None:
        super(VAE, self).__init__()

        self.latent_dim2 = latent_dim2
        self.cloud_weight = cloud_weight
        self.kl_weight1 = kl_weight1
        self.kl_weight2 = kl_weight2
        self.param_weight = param_weight
        self.arch_option = arch_option
        self.lstm_option = lstm_option
        self.data_path = data_path
        self.temporal_resolution = temporal_resolution

        with h5py.File(os.path.join(self.data_path, 'dataset_T='+str(self.temporal_resolution)+'.h5'), 'r') as hf:
            self.mean = torch.tensor(hf['mean_train'][:], dtype=torch.float32)
            self.var = torch.tensor(hf['variance_train'][:], dtype=torch.float32)

        # Define the activations
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

        # Build Encoder
        if self.temporal_resolution == 10:
            self.conv1  = nn.LazyConv3d(32, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        elif self.temporal_resolution == 20:
            self.conv1  = nn.LazyConv3d(32, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        else:
            raise ValueError(f"Invalid temporal_resolution: {self.temporal_resolution}. Expected 10 or 20.")
        self.conv2  = nn.LazyConv3d(64, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3  = nn.LazyConv3d(128, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.conv33 = nn.LazyConv3d(128, (3, 3, 3), padding=(1, 1, 1))
        #(N, C, D, H, W)

        if (self.arch_option == 'spatial_attention'):
            self.encoder_attention = SpatialAttention(128) # Assuming 128 channels in conv33
            self.decoder_attention = SpatialAttention(128) # Assuming 128 channels in result

        if (self.lstm_option == 'temporal_attention'):
            self.temporal_attention1 = TemporalAttention(256)
            self.temporal_attention2 = TemporalAttention(256)
            self.temporal_attention3 = TemporalAttention(256)
            self.temporal_attention4 = TemporalAttention(256)

        self.lstm_layer = nn.LSTM(input_size=32768, hidden_size=2048, num_layers=1, bidirectional=False, batch_first=True)

        self.lstm_layer1 = nn.LSTM(input_size=2048, hidden_size=256, num_layers=1, bidirectional=False, batch_first=True)
        self.lstm_layer2 = nn.LSTM(input_size=2048, hidden_size=256, num_layers=1, bidirectional=False, batch_first=True)
        self.lstm_layer3 = nn.LSTM(input_size=2048, hidden_size=256, num_layers=1, bidirectional=False, batch_first=True)

        self.lstm_layer_new = nn.LSTM(input_size=2048, hidden_size=256, num_layers=1, bidirectional=False, batch_first=True)
        self.mean_new = nn.LazyLinear(self.latent_dim2)
        self.var_new = nn.LazyLinear(self.latent_dim2)

        self.mean1 = nn.LazyLinear(2)
        self.var1 = nn.LazyLinear(2)

        self.mean2 = nn.LazyLinear(1)
        self.var2 = nn.LazyLinear(1)

        self.mean3 = nn.LazyLinear(2)
        self.var3 = nn.LazyLinear(2)

        # Build Decoder
        self.dense = nn.LazyLinear(128*5*16*16)

        # Define the expansive path of the network
        self.up4 = nn.LazyConvTranspose3d(128, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)) ################
        self.up5 = nn.LazyConvTranspose3d(64, (4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        if self.temporal_resolution == 10:
            self.up6 = nn.LazyConvTranspose3d(32, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        elif self.temporal_resolution == 20:
            self.up6 = nn.LazyConvTranspose3d(32, (4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        else:
            raise ValueError(f"Invalid temporal_resolution: {self.temporal_resolution}. Expected 10 or 20.")

        # Define the output layer of the network
        self.output_layer = nn.LazyConv3d(1, (1, 1, 1), padding=(0, 0, 0))


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Forward the input by passing through the encoder-decoder network
        :param input: (Tensor) Input tensor to encoder [N x C x D x H x W]
        :return: (Tensor) List of outputs
        """
        conv1 = self.relu(self.conv1(input)) # its shape is (batch_size, channels, depth, height, width)
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        if (self.arch_option == 'spatial_attention'):
            conv3 = self.encoder_attention(conv3) # Apply attention in encoder
        conv33 = self.relu(self.conv33(conv3))
    
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
        lstm_new, _ = self.lstm_layer_new(lstm)

        if (self.lstm_option == 'temporal_attention'):
            lstm1 = self.temporal_attention1(lstm1)
            lstm2 = self.temporal_attention2(lstm2)
            lstm3 = self.temporal_attention3(lstm3)
            lstm_new = self.temporal_attention4(lstm_new)
        else:
            lstm1 = lstm1[:,-1,:]
            lstm2 = lstm2[:,-1,:]
            lstm3 = lstm3[:,-1,:]
            lstm_new = lstm_new[:,-1,:]

        mean_new = self.mean_new(lstm_new)
        var_new = self.softplus(self.var_new(lstm_new))
        z_new = self.reparameterize(mean_new, var_new)

        mean1 = self.mean1(lstm1)
        var1 = self.softplus(self.var1(lstm1))
        z1 = self.reparameterize(mean1, var1)

        mean2 = self.mean2(lstm2)
        var2 = self.softplus(self.var2(lstm2))
        z2 = self.reparameterize(mean2, var2)

        mean3 = self.mean3(lstm3)
        var3 = self.softplus(self.var3(lstm3))
        z3 = self.reparameterize(mean3, var3)

        concatted = torch.cat((z1, z2, z3, z_new), dim=1)
        result = self.dense(concatted)
        result = result.view(-1, conv33.shape[1], conv33.shape[2], conv33.shape[3], conv33.shape[4])

        up4 = self.relu(self.up4(result))
        if (self.arch_option == 'spatial_attention'):
            up4 = self.decoder_attention(up4) # Apply attention in decoder

        up5 = self.relu(self.up5(up4)) 
        up6 = self.relu(self.up6(up5))
        output = self.output_layer(up6)

        return [output, mean1, var1, z1, mean2, var2, z2, mean3, var3, z3, mean_new, var_new, z_new]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick
        """
        # std = torch.exp(0.5 * logvar)
        std = torch.sqrt(logvar + 1e-6) #This is var indeed not logvar
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def kl_divergence_two_gauss(self, mean1, var1):
        curr_device = mean1.device
        self.mean = self.mean.cuda(curr_device)
        self.var = self.var.cuda(curr_device)
        kl1 = torch.log(torch.sqrt(self.var)) - torch.log(torch.sqrt(var1)) + ((var1 + torch.square(mean1-self.mean)) / (2*self.var)) - 0.5
        kl2 = torch.sum(kl1, dim=1)
        kl = torch.mean(kl2)
        return kl

    # divergence = -1/2 * sum( 1 + log(sigma^2) - sigma^2 - mean^2 )
    def kl_divergence(self, mean, var):
        return -0.5 * torch.sum(1 + torch.log(var) - var - torch.square(mean), dim=1)


    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        """
        cloud, params = kwargs['inputs']
        recons = args[0]

        mean1 = args[1]
        var1 = args[2]
        z1 = args[3]

        mean2 = args[4]
        var2 = args[5]
        z2 = args[6]

        mean3 = args[7]
        var3 = args[8]
        z3 = args[9]

        cloud_loss =F.mse_loss(recons, cloud)
            
        xy_loss = F.mse_loss(z1, params[:,:2])
        t_loss = F.mse_loss(z2, params[:,2:3])
        uv_loss = F.mse_loss(z3, params[:,3:])

        mean = torch.cat((mean1, mean2, mean3), dim=1)
        var = torch.cat((var1, var2, var3), dim=1)
        kl_loss = self.kl_divergence_two_gauss(mean, var)

        mean_new = args[10]
        var_new = args[11]
        z_new = args[12]
        kl_loss_new = torch.mean(self.kl_divergence(mean_new, var_new))
        loss = self.param_weight*(xy_loss + t_loss + uv_loss) + self.kl_weight2*kl_loss_new + self.kl_weight1*kl_loss + self.cloud_weight*cloud_loss
        
        return {'loss': loss, 'Cloud_Loss':cloud_loss.detach(), 'KL':kl_loss.detach(), 'xy':xy_loss.detach(), 't':t_loss.detach(), 'uv':uv_loss.detach()}

    def encode(self, input: Tensor, **kwargs) -> List[Tensor]:

        conv1 = self.relu(self.conv1(input)) # its shape is (batch_size, channels, depth, height, width)
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        if (self.arch_option == 'spatial_attention'):
            conv3 = self.encoder_attention(conv3) # Apply attention in encoder
        conv33 = self.relu(self.conv33(conv3))
    
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
        lstm_new, _ = self.lstm_layer_new(lstm)

        if (self.lstm_option == 'temporal_attention'):
            lstm1 = self.temporal_attention1(lstm1)
            lstm2 = self.temporal_attention2(lstm2)
            lstm3 = self.temporal_attention3(lstm3)
            lstm_new = self.temporal_attention4(lstm_new)
        else:
            lstm1 = lstm1[:,-1,:]
            lstm2 = lstm2[:,-1,:]
            lstm3 = lstm3[:,-1,:]
            lstm_new = lstm_new[:,-1,:]

        mean_new = self.mean_new(lstm_new)
        var_new = self.softplus(self.var_new(lstm_new))
        z_new = self.reparameterize(mean_new, var_new)

        mean1 = self.mean1(lstm1)
        var1 = self.softplus(self.var1(lstm1))
        z1 = self.reparameterize(mean1, var1)

        mean2 = self.mean2(lstm2)
        var2 = self.softplus(self.var2(lstm2))
        z2 = self.reparameterize(mean2, var2)

        mean3 = self.mean3(lstm3)
        var3 = self.softplus(self.var3(lstm3))
        z3 = self.reparameterize(mean3, var3)

        return [conv33, mean1, var1, z1, mean2, var2, z2, mean3, var3, z3, mean_new, var_new, z_new]



    def decode(self, input: Tensor, **kwargs) -> List[Tensor]:
        z1 = input[0]
        z2 = input[1]
        z3 = input[2]
        z_new = input[3]

        concatted = torch.cat((z1, z2, z3, z_new), dim=1)
        result = self.dense(concatted)
        result = result.view(-1, 128, 5, 16, 16)

        up4 = self.relu(self.up4(result))
        if (self.arch_option == 'spatial_attention'):
            up4 = self.decoder_attention(up4) # Apply attention in decoder

        up5 = self.relu(self.up5(up4)) 
        up6 = self.relu(self.up6(up5))
        output = self.output_layer(up6)  

        return output





