import torch
import torch.nn as nn
import torchaudio
from torchaudio.prototype.functional import sinc_impulse_response
from utils import utils


class RNN(nn.Module):
    def __init__(self, input_size=512, hidden_size=64, output_size=512, num_layers=2):
        super(RNN, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        

    def encoderDecoder(self, x):
        # Encoder
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(x)
        # Decoder
        decoder_output, _ = self.decoder(encoder_output, (encoder_hidden, encoder_cell))
        # Output layer
        x = self.output_layer(decoder_output) #(n, 2, 512)
        
        return x

    def forward(self, x):
        '''
            x: (n, 512) # n is batch size
            return: (n, 512)
        '''
        x = torch.unsqueeze(x, dim = 1) #output: (n, 1, 512)

        x = self.encoderDecoder(x) #(n, 1, 512)
        print(x.shape)
        return x
