import json
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.mydataset import MyCustomDataset
import random
import parameters.xela_params as xela_params
from preprocessing.preprocessing_functions import filter_tactile2, filter_force


def to_tensor(data):
    '''
    This function converts the input data to a PyTorch tensor.

    Args:
        data (np.ndarray): The input data.

    Returns:
        torch.Tensor: The PyTorch tensor.
    '''

    return torch.tensor(data, dtype=torch.float32)


def generate_unique_random_integers(n, min, max):
    '''
    This function generates n unique random integers in the range [min, max].
    Args:
        n: int, the number of unique random integers to generate.
        min: int, the minimum value of the range.
        max: int, the maximum value of the range.
    Returns:
        list, the list of unique random integers.
    '''

    if max - min + 1 < n:
        raise ValueError("Range of numbers is smaller than required unique numbers")
    return random.sample(range(min, max), n)


def get_tip_datapath(n_train, n_val, finger):
    '''
    This function returns the paths of the training and validation datasets for a given finger.
    The datasets are randomly selected from the total number of datasets available for the finger.
    The training and validation datasets are disjoint.
    Args:
        n_train: int, the number of training datasets.
        n_val: int, the number of validation datasets.
        finger: str, the name of the finger.
    Returns:
        train: list, the paths of the training datasets.
        val: list, the paths of the validation datasets.
    '''

    data_dict = json.load(open('dataset/datasets.json'))
    datasets =  data_dict["tips"][finger]
    n_tot = len(datasets)

    if n_train + n_val > n_tot:
        raise ValueError('n_train + n_val must be less than or equal to the total number of datasets for the selected finger')
    
    idx_list = generate_unique_random_integers(n_train+n_val, 0, n_tot)
    train = [datasets[i] for i in idx_list[:n_train]]
    val = [datasets[i] for i in idx_list[n_train:]]
    return train, val

def get_phal_datapath(n_train, n_val, finger):
    '''
    This function returns the paths of the training and validation datasets for a given finger.
    The datasets are randomly selected from the total number of datasets available for the finger.
    The training and validation datasets are disjoint.
    Args:
        n_train: int, the number of training datasets.
        n_val: int, the number of validation datasets.
        finger: str, the name of the finger.
    Returns:
        train: list, the paths of the training datasets.
        val: list, the paths of the validation datasets.
    '''

    data_dict = json.load(open('dataset/datasets.json'))
    datasets =  data_dict["phal"][finger]
    n_tot = len(datasets)

    if n_train + n_val > n_tot:
        raise ValueError('n_train + n_val must be less than or equal to the total number of datasets for the selected finger')
    
    idx_list = generate_unique_random_integers(n_train+n_val, 0, n_tot)
    train = [datasets[i] for i in idx_list[:n_train]]
    if n_val:
        val = [datasets[i] for i in idx_list[n_train:]]
        return train, val
    else:
        return train


def load_data(folder_path, metadata=False):
    '''
    This function loads the data from the specified folder path.
    Args:
        folder_path: str, the path of the folder containing the data.
        metadata: bool, if True, the function returns the metadata along with the inputs and labels.
    Returns:
        x: torch.tensor, the input data.
        y: torch.tensor, the label data.
        p: dict, the metadata.
    '''

    if metadata:
        p = json.load(open(folder_path + 'metadata.json'))

    inputs_path = folder_path + 'tact_data.pkl'
    labels_path = folder_path + 'force_label.pkl'
    time_path = folder_path + 'time.pkl'

    with open(inputs_path, 'rb') as f:
        x = pickle.load(f)

    with open(labels_path, 'rb') as f:
        y = pickle.load(f)

    with open(time_path, 'rb') as f:
        t = pickle.load(f)

    x = filter_tactile2(t, x)
    y = filter_force(t, y)

    x = to_tensor(x)
    y = to_tensor(y)

    if metadata:
        return x, y, p
    else:
        return x, y
    

def load_concat_data(folder_path_list):
    '''
    This function loads the data from the specified folder paths and concatenates them.
    Args:
        folder_path_list: list, the list of folder paths containing the data.
    Returns:
        x_concat: torch.tensor, the concatenated input data.
        y_concat: torch.tensor, the concatenated label data.
    '''

    i = 0
    for folder_path in folder_path_list:
        x, y = load_data(folder_path)
        if i:
            x_concat = torch.cat((x_concat, x), 0)
            y_concat = torch.cat((y_concat, y), 0)
        else:
            x_concat = x
            y_concat = y
            i = 1
    return x_concat, y_concat


def avgpool_dataset(x, y, window_size=10):
    '''
    This function averages the input and label data over a window of size window_size.
    Args:
        x: torch.tensor, the input data.
        y: torch.tensor, the label data.
        window_size: int, the size of the window.
    Returns:
        x_ds: torch.tensor, the downsampled input data.
        y_ds: torch.tensor, the downsampled label data.
    '''
    x = x.reshape(x.shape[0], -1)
    x_ds = F.avg_pool1d(x.permute(1,0).unsqueeze(0), window_size).squeeze(0)
    y_ds = F.avg_pool1d(y.permute(1,0).unsqueeze(0), window_size).squeeze(0)
    x = x_ds.permute(1, 0)
    y = y_ds.permute(1, 0)
    x = x.reshape((x.shape[0], -1, 3))
    return x, y


def reshape_imgs(x, size, _mode='bilinear'):
    '''
    This function reshapes the input tensor to a different size.
    Args:
        x: torch.tensor, the input tensor.
        size: tuple, the new size of the tensor.
        _mode: str, the interpolation mode used in the reshaping.
    Returns:
        x: torch.tensor, the reshaped tensor.
    '''
    x = x.permute(0, 3, 1, 2)  # Convert from (N, 6, 6, 3) to (N, 3, 6, 6)
    return F.interpolate(x, size=(size[0], size[1]), mode=_mode, align_corners=False)


def to_tip_imgs(x, reshape=False, size=None, mode='bilinear'):
    '''
    This funtions takes a tensor of shape (N, 90) and returns a tensor of shape (N, 3, 6, 6).
    The output tensor can also be reshaped to a different img size.
    Args:
        x: torch.tensor of shape (N, 90)
        reshape: bool, if True, reshapes the output tensor to a different size
        size: tuple, the new size of the output tensor
        mode: str, the interpolation mode used in the reshaping
    Returns:
        imgs: The output tensor.
    '''
    x_dim = x[:, :, 0]
    y_dim = x[:, :, 1]
    z_dim = x[:, :, 2]
    len = x.shape[0]
    n_taxels = x.shape[1]
    if n_taxels == xela_params.N_TAXEL_TIP:
        x_img = torch.cat((torch.zeros(len,2), x_dim[:,0:4], torch.zeros(len,1), x_dim[:,4:21],
            torch.zeros(len, 1), x_dim[:, 21:26], torch.zeros(len, 2), x_dim[:,26:]), dim=1).reshape(len, 6, 6)
        y_img = torch.cat((torch.zeros(len,2), y_dim[:,0:4],
                            torch.zeros(len,1), y_dim[:,4:21],
                            torch.zeros(len, 1), y_dim[:, 21:26], 
                            torch.zeros(len, 2), y_dim[:,26:]), dim=1).reshape(len, 6, 6)
        z_img = torch.cat((torch.zeros(len,2), z_dim[:,0:4],
                            torch.zeros(len,1), z_dim[:,4:21],
                            torch.zeros(len, 1), z_dim[:, 21:26], 
                            torch.zeros(len, 2), z_dim[:,26:]), dim=1).reshape(len, 6, 6)
    elif n_taxels == xela_params.N_TAXEL_PHAL:
        x_img = x_dim.reshape(len, 4, 4)
        y_img = y_dim.reshape(len, 4, 4)
        z_img = z_dim.reshape(len, 4, 4)
    
    imgs =  torch.stack((x_img, y_img, z_img), dim=-1)

    if reshape:
        return reshape_imgs(imgs, size, mode)
    else:
        return imgs.permute(0, 3, 1, 2)


def create_data_loader(folder_path_list, batch_size, shuffle, regularize=False, as_img=False, reshape=False, size=None, mode='bilinear',avg_pool=False, avg_pool_size=2):
    '''
    This function creates a PyTorch DataLoader object from the input data.
    Args:
        folder_path_list: list, the list of folder paths containing the data.
        batch_size: int, the batch size.
        shuffle: bool, if True, the data is shuffled.
        as_img: bool, if True, the input data is converted to images.
        reshape: bool, if True, the output images are reshaped.
        size: tuple, the new size of the output images.
        mode: str, the interpolation mode used in the reshaping.
        avg_pool: bool, if True, the input data is averaged over a window.
        avg_pool_size: int, the size of the window.
    Returns:
        DataLoader: The PyTorch DataLoader object.
    '''

    inputs, labels = load_concat_data(folder_path_list)
    if avg_pool:
        print(inputs.shape)
        inputs, labels = avgpool_dataset(inputs, labels, avg_pool_size)
        print(inputs.shape)
    shape = 'tab'
    if as_img:
        print("before to img: ", inputs.shape)
        inputs = to_tip_imgs(inputs, reshape, size, mode)
        shape = 'img'
    dataset = MyCustomDataset(inputs, labels, shape)
    if regularize:
        params = dataset.standardize_data()
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), params
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def create_data_loader2(folder_path_list, batch_size, shuffle, as_img=False, reshape=False, size=None, mode='bilinear',avg_pool=False, avg_pool_size=2, seq=False, seq_len=10):
    '''
    This function creates a PyTorch DataLoader object from the input data.
    Args:
        folder_path_list: list, the list of folder paths containing the data.
        batch_size: int, the batch size.
        shuffle: bool, if True, the data is shuffled.
        as_img: bool, if True, the input data is converted to images.
        reshape: bool, if True, the output images are reshaped.
        size: tuple, the new size of the output images.
        mode: str, the interpolation mode used in the reshaping.
        avg_pool: bool, if True, the input data is averaged over a window.
        avg_pool_size: int, the size of the window.
    Returns:
        DataLoader: The PyTorch DataLoader object.
    '''

    inputs, labels = load_concat_data(folder_path_list)
    if avg_pool:
        inputs, labels = avgpool_dataset(inputs, labels, avg_pool_size)
    if as_img:
        inputs = to_tip_imgs(inputs, reshape, size, mode)
    if seq:
        inputs, labels = to_sequences(inputs, labels, seq_len)
    dataset = MyCustomDataset(inputs, labels)
    params = dataset.standardize_data()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), params

def to_sequences(X, Y, seq_len):
    '''
    This function converts the input and label data to sequences of length seq_len.
    Args:
        X: torch.tensor, the input data.
        Y: torch.tensor, the label data.
        seq_len: int, the length of the sequences.
    Returns:
        X_seq: torch.tensor, the input sequences.
        Y_seq: torch.tensor, the label sequences.
    '''

    shape_x = X.shape
    shape_y = Y.shape

    n_seq = shape_x[0] // seq_len
    new_len = shape_x[0] - shape_x[0] % seq_len

    new_shape_X = (n_seq, seq_len) + shape_x[1:]
    new_shape_Y = (n_seq, seq_len) + shape_y[1:]

    X_seq = X[:new_len].reshape(new_shape_X)
    Y_seq = Y[:new_len].reshape(new_shape_Y)

    return X_seq, Y_seq