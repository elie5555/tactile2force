import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    '''
    This class defines a fully connected neural network with 3 hidden layers.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer with 90 input neurons and 64 output neurons.
        fc2 (nn.Linear): The second fully connected layer with 64 input neurons and 32 output neurons.
        fc3 (nn.Linear): The third fully connected layer with 32 input neurons and 3 output neurons.

    Methods:
        forward(x): Forward pass of the neural network.
        save_weights(filepath): Save the weights of the neural network to a file.
    '''

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(270, 128) # Input layer: 90 neurons, Hidden layer: 64 neurons
        self.fc2 = nn.Linear(128, 64)  # Hidden layer: 128 neurons, Hidden layer: 64 neurons
        self.fc3 = nn.Linear(64, 32)   # Hidden layer: 64 neurons, Hidden layer: 32 neurons
        self.fc4 = nn.Linear(32, 3)    # Hidden layer: 32 neurons, Output layer: 3 neurons
        #nn.BatchNorm1d
    def forward(self, x):
        x = torch.relu(self.fc1(x))    # ReLU activation function for the first hidden layer
        x = torch.relu(self.fc2(x))    # ReLU activation function for the second hidden layer
        x = torch.relu(self.fc3(x))    # ReLU activation function for the third hidden layer
        x = self.fc4(x)                # Output layer (no activation function applied)
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

def to_tensor(data):
    '''
    This function converts the input data to a PyTorch tensor.

    Args:
        data (np.ndarray): The input data.

    Returns:
        torch.Tensor: The PyTorch tensor.
    '''

    return torch.tensor(data, dtype=torch.float32)


def split_data(train_data, train_labels, test_ratio=0.2, random_state=None):
    '''
    This function splits the training data into training and testing sets.

    Args:
        train_data (np.ndarray): The training data.
        train_labels (np.ndarray): The training labels.
        test_ratio (float): The ratio of the data to be used for testing.
        random_state (int): The random seed for reproducibility.

    Returns:
        np.ndarray: The training data.
        np.ndarray: The testing data.
        np.ndarray: The training labels.
        np.ndarray: The testing labels.
    '''

    np.random.seed(random_state)
    indices = np.random.permutation(len(train_data))
    num_test = int(len(train_data) * test_ratio)
    train_indices, test_indices = indices[num_test:], indices[:num_test]
    return train_data[train_indices], train_data[test_indices], train_labels[train_indices], train_labels[test_indices]

def train_network(model, data_tensor, labels_tensor, num_epochs=60, learning_rate=0.00025):
    '''
    This function trains the neural network model.

    Args:
        model (NeuralNetwork): The neural network model to be trained.
        train_data (np.ndarray): The training data.
        train_labels (np.ndarray): The training labels.
        num_epochs (int): The number of epochs for training.
        learning_rate (float): The learning rate for the optimizer.
    
    Returns:
        list: The training loss for each epoch.
    '''

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # data_tensor = torch.tensor(train_data, dtype=torch.float32)
    # labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
    
    train_loss = []
    validation_loss = []

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        # Split data into training and validation sets
        train_data_tensor, validation_data_tensor, train_labels_tensor, validation_data_labels  = split_data(data_tensor, labels_tensor, test_ratio=0.1)
        
        # Training loop for each batch
        for inputs, labels in zip(train_data_tensor, train_labels_tensor):
            optimizer.zero_grad()
            inputs = inputs.flatten()
            outputs = model(inputs.unsqueeze(0))
            loss = criterion(outputs, labels.unsqueeze(0))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation loss
        l, _ = test_network(model, validation_data_tensor, validation_data_labels)
        validation_loss.append(l)
        epoch_loss = running_loss / len(data_tensor)
        train_loss.append(epoch_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training loss: {epoch_loss}, Validation loss: {l}')

    plt.plot(train_loss, label='training loss')
    plt.plot(validation_loss, label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Force [N]')
    plt.legend()
    plt.show()

def test_network(model, test_data_tensor, test_labels_tensor):
    '''
    This function evaluates the neural network model on the test data.

    Args:
        model (NeuralNetwork): The trained neural network model.
        test_data (np.ndarray): The test data.
        test_labels (np.ndarray): The test labels.

    Returns:
        float: The test loss.
        np.ndarray: The predictions of the model on the test data.
    '''

    model.eval()

    # test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    # test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(test_data_tensor)
    
    criterion = nn.MSELoss()
    test_loss = criterion(predictions, test_labels_tensor).item()
    return test_loss, predictions.numpy()