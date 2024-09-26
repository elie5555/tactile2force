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
        #self.fc1 = nn.Linear(90, 128)  # Input layer: 90 neurons, Hidden layer: 128 neurons
        #self.fc2 = nn.Linear(128, 64)  # Hidden layer: 128 neurons, Hidden layer: 64 neurons
        self.fc1 = nn.Linear(90, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        #x = torch.nn.functional.relu(self.fc1(x))    # ReLU activation function for the first hidden layer
        #x = torch.nn.functional.relu(self.fc2(x))    # ReLU activation function for the second hidden layer

        x = torch.relu(self.fc1(x))    # ReLU activation function for the first hidden layer
        x = torch.relu(self.fc2(x))    # ReLU activation function for the second hidden layer
        #x = torch.relu(self.fc3(x))    # ReLU activation function for the third hidden layer
        x = self.fc3(x)                # Output layer (no activation function applied)
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


def train_network(model, training_loader, validation_loader, num_epochs=60, learning_rate=0.00025):
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
        
        # Training loop for each batch
        for i, data in enumerate(training_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


        # epoch loss
        epoch_loss = running_loss / (i+1)
        train_loss.append(epoch_loss)  
        
        # Validation loss
        val_loss = test_network(model, validation_loader)
        validation_loss.append(val_loss)

        if epoch % 20 == 19:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training loss: {epoch_loss}, Validation loss: {val_loss}')

    plt.plot(train_loss, label='training loss')
    plt.plot(validation_loss, label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Force [N]')
    plt.legend()
    plt.show()
    

def test_network(model, validation_loader):
    '''
    This function evaluates the neural network model on the validation data.

    Args:
        model (NeuralNetwork): The trained neural network model.
        validation_loader (DataLoader): The validation data loader.

    Returns:
        float: The validation loss.
    '''

    model.eval()
    criterion = nn.MSELoss()
    running_vloss = 0.0

    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss.item()  # Convert tensor to scalar and add to running loss

    avg_vloss = running_vloss / (i + 1)
    #print('LOSS valid {}'.format(avg_vloss))
    return avg_vloss


def predict(model, inputs):
    model.eval()
    return model(inputs)
