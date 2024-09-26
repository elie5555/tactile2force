import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    '''
    This class defines a fully connected autoencoder with 3 hidden layers.

    Attributes:
        encoder (nn.Sequential): The encoder part of the autoencoder.
        decoder (nn.Sequential): The decoder part of the autoencoder.

    Methods:
        forward(x): Forward pass of the autoencoder.
        save_weights(filepath): Save the weights of the autoencoder to a file.
    '''

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(90, 64),  # Input layer: 90 neurons, Hidden layer: 64 neurons
            nn.ReLU(),          # ReLU activation function
            nn.Linear(64, 32),  # Hidden layer: 64 neurons, Hidden layer: 32 neurons
            nn.ReLU(),          # ReLU activation function
            nn.Linear(32, 16),  # Hidden layer: 32 neurons, Hidden layer: 16 neurons
            nn.ReLU()           # ReLU activation function
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),  # Input layer: 16 neurons, Hidden layer: 32 neurons
            nn.ReLU(),          # ReLU activation function
            nn.Linear(32, 64),  # Hidden layer: 32 neurons, Hidden layer: 64 neurons
            nn.ReLU(),          # ReLU activation function
            nn.Linear(64, 90),  # Hidden layer: 64 neurons, Output layer: 90 neurons
            nn.Sigmoid()        # Sigmoid activation function
        )

        

    def forward(self, x):
        x = self.encoder(x)  # Encoder pass
        x = self.decoder(x)  # Decoder pass
        return x

    def save_weights(self, filepath):
        # Check if the file already exists
        if os.path.exists(filepath):
            # If the file exists, modify the filepath to avoid overwriting
            base, ext = os.path.splitext(filepath)
            filepath = base + "_new" + ext

        # Get the state dictionary of the model's weights
        state_dict = self.state_dict()
        # Save the state dictionary to a file using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(state_dict, f)