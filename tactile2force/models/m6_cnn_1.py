import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

class CNN2(nn.Module):
    def __init__(self, patch_type='tip'):
        super(CNN2, self).__init__()
        
        if patch_type == 'tip':
            input_size = (6,6)
        elif patch_type == 'phal':
            input_size = (4,4)
        
        self.input_size = input_size
        self.patch_type = patch_type

        # Convolutional layers
        _kernel_size = 3
        _stride=1
        _padding=0
        max_depth = 24

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        out_h = math.floor(self.input_size[0]+2*_padding-(_kernel_size-1)-1)/_stride + 1
        out_w = math.floor(self.input_size[1]+2*_padding-(_kernel_size-1)-1)/_stride + 1

        if patch_type == 'tip':
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=0)
            out_h = math.floor(out_h+2*_padding-(_kernel_size-1)-1)/_stride + 1
            out_w = math.floor(out_w+2*_padding-(_kernel_size-1)-1)/_stride + 1

            self.batch_norm = nn.BatchNorm2d(12)
        else:
            self.batch_norm = nn.BatchNorm2d(6)
            
        if patch_type == 'tip':
            self.conv3 = nn.Conv2d(in_channels=12, out_channels=max_depth, kernel_size=2, stride=1, padding=0)
            out_h = math.floor(out_h+2*_padding-(2-1)-1)/_stride + 1
            out_w = math.floor(out_w+2*_padding-(2-1)-1)/_stride + 1
        else:
            self.conv3 = nn.Conv2d(in_channels=6, out_channels=max_depth, kernel_size=2, stride=1, padding=0)
            out_h = math.floor(out_h+2*_padding-(2-1)-1)/_stride + 1
            out_w = math.floor(out_w+2*_padding-(2-1)-1)/_stride + 1

        self.conv_out_h = out_h
        self.conv_out_w = out_w
        self.conv_out_depth = max_depth
        
        # Fully connected layers
        self.n_fc1 = int(max_depth * out_h * out_w)
        self.n_fc_2 = 16
        self.n_out = 3
        self.fc1 = nn.Linear(self.n_fc1, self.n_fc_2)
        self.fc2 = nn.Linear(self.n_fc_2, self.n_out)
        
    def forward(self, x):

        x = F.relu(self.conv1(x))
        if self.patch_type == 'tip':
            x = F.relu(self.conv2(x))
        x = self.batch_norm(x)
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = x.contiguous().view(-1, self.n_fc1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def to_tensor(data):
    '''
    This function converts the input data to a PyTorch tensor.

    Args:
        data (np.ndarray): The input data.

    Returns:
        torch.Tensor: The PyTorch tensor.
    '''
    return torch.tensor(data, dtype=torch.float32)


def train_network(model, training_loader, validation_loader, num_epochs=60, learning_rate=0.00025, autoclose=True):
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
            # print("input shape in training loop: ", inputs.shape)
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

        if epoch % 10 == 9:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training loss: {epoch_loss}, Validation loss: {val_loss}')

    plt.plot(train_loss, label='training loss')
    plt.plot(validation_loss, label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Force [N]')
    plt.legend()
    if autoclose:
        plt.show(block=False)
        plt.close()
    else:
        plt.show()
    return train_loss, val_loss
    

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