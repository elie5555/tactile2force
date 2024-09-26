import torch
from torch.utils.data import Dataset, DataLoader

class MyCustomDataset(Dataset):
    def __init__(self, data, labels, _shape):
        '''
        This class is a custom dataset class that inherits from the PyTorch Dataset class.
        Args:
            data (torch.Tensor): Tensor containing the data.
            labels (torch.Tensor): Tensor containing the labels.
            _shape (str): Shape of the data (img or tab)
        '''
        self.data = data
        self.labels = labels
        self.shape = _shape
    
    def __len__(self):
        '''
        Returns the length of the dataset
        Args:
            None
        Returns:   
            int: Length of the dataset
        '''
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        Retrieves the sample and its corresponding label
        Args:
            idx (int): Index of the sample to be retrieved
        Return:
            tuple: Tuple containing the sample and its corresponding label
        '''
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
    
    def _get_std_dict(self):
        '''
        Computes stadardization parameters from data and label already contained in the dataset
        Args:
            None
        Returns:
            dict: Dictionary containing the standardization parameters
        '''
        if self.shape == 'img':
            std_data = torch.tensor([self.data[:, 0, :, :].std(), self.data[:, 1, :, :].std(), self.data[:, 2, :, :].std()])
        elif self.shape == 'tab':
            std_data = torch.tensor([self.data[:, :, 0].std(), self.data[:, :, 1].std(), self.data[:, :, 2].std()])
        
        std_labels = self.labels.std(axis=0)
        return {"tactile_scale" : std_data.tolist(), "force_scale" : std_labels.tolist()}
    
    def standardize_label_ext(self, std_dict):
        '''
        Standardizes the labels of the dataset from given standardization parameters
        Args:
            std_dict (dict): Dictionary containing the standardization parameters
        Returns:
            None
        '''
        std_labels = std_dict["force_scale"]
        std_labels = self.labels.std(axis=0)
        self.labels = self.labels / std_labels

    def standardize_data_ext(self, std_dict):
        '''
        Standardizes the data of the dataset from given standardization parameters
        Args:
            std_dict (dict): Dictionary containing the standardization parameters
        Returns:
            None
        '''
        std_data = std_dict["tactile_scale"]
        if self.shape == 'img':
            self.data[:,0,:,:] = self.data[:,0,:,:] / std_data[0]
            self.data[:,1,:,:] = self.data[:,1,:,:] / std_data[1]
            self.data[:,2,:,:] = self.data[:,2,:,:] / std_data[2]

        elif self.shape == 'tab':
            self.data[:,:,0] = self.data[:,:,0] / std_data[0]
            self.data[:,:,1] = self.data[:,:,1] / std_data[1]
            self.data[:,:,2] = self.data[:,:,2] / std_data[2]
    
    def apply_standardization(self, std_dict):
        '''
        Applies standardization to the dataset
        Args:
            std_dict (dict): Dictionary containing the standardization parameters
        Returns:
            None
        '''
        self.standardize_data_ext(std_dict)
        self.standardize_label_ext(std_dict)

    def standardize_data(self):
        '''
        Standardizes the data and labels of the dataset from the dataset itself and returns the standardization parameters
        Args:
            None
        Returns:
            dict: Dictionary containing the standardization parameters
        '''
        std_dict = self._get_std_dict()
        self.apply_standardization(std_dict)
        return std_dict
    
    def reshape_to_vect(self):
        self.data = self.data.reshape((self.data.shape[0], -1))