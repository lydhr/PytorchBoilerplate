import torch
import torch.nn as nn
from utils import utils

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding='valid'), #output: (n, 1, 2047)
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2), #(n, 1, 1023)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(510, 128),
            nn.Linear(128, 2),
            nn.Softmax(dim=1),
        )

        
    def forward(self, x):
        '''
            x: (n, 512) # n is batch size
            return: (n, 2)
        '''
        x = torch.unsqueeze(x, dim = 1) #output: (n, 1, 2048)
        x = self.cnn_layers(x)
        x = self.cnn_layers(x) #(n, 1, x)
        x = x.view(x.size(0), -1) #(n, x)
        x = self.linear_layers(x) #(n, 2)
        return x