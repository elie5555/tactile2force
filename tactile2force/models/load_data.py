import models.fully_connected_network as fcn
import json
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def load_data(folder_path, metadata=False):
    if metadata:
        p = json.load(open(folder_path + 'metadata.json'))

    inputs_path = folder_path + 'tact_data.pkl'
    labels_path = folder_path + 'force_label.pkl'

    with open(inputs_path, 'rb') as f:
        x = fcn.to_tensor(pickle.load(f))

    with open(labels_path, 'rb') as f:
        y = fcn.to_tensor(pickle.load(f))

    if metadata:
        return x, y, p
    else:
        return x, y
    
def load_concat_data(folder_path_list):
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

def load_metadata(folder_path):
    p = json.load(open(folder_path + 'metadata.json'))
    return p

def avgpool_dataset(x, y, window_size=10):
    x_ds = F.avg_pool1d(x.permute(1,0).unsqueeze(0), window_size).squeeze(0)
    y_ds = F.avg_pool1d(y.permute(1,0).unsqueeze(0), window_size).squeeze(0)
    return x_ds.permute(1, 0), y_ds.permute(1, 0)

def reshape_imgs(x, size, _mode='bilinear'):
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
    x_dim = x[:, 0:30]
    y_dim = x[:, 30:60]
    z_dim = x[:, 60:90]
    len = x.shape[0]
    x_img = torch.cat((torch.zeros(len,2), x_dim[:,0:4],
                        torch.zeros(len,1), x_dim[:,4:21],
                        torch.zeros(len, 1), x_dim[:, 21:26], 
                        torch.zeros(len, 2), x_dim[:,26:]), dim=1).reshape(len, 6, 6)
    y_img = torch.cat((torch.zeros(len,2), y_dim[:,0:4],
                        torch.zeros(len,1), y_dim[:,4:21],
                        torch.zeros(len, 1), y_dim[:, 21:26], 
                        torch.zeros(len, 2), y_dim[:,26:]), dim=1).reshape(len, 6, 6)
    z_img = torch.cat((torch.zeros(len,2), z_dim[:,0:4],
                        torch.zeros(len,1), z_dim[:,4:21],
                        torch.zeros(len, 1), z_dim[:, 21:26], 
                        torch.zeros(len, 2), z_dim[:,26:]), dim=1).reshape(len, 6, 6)
    
    imgs =  torch.stack((x_img, y_img, z_img), dim=-1)

    if reshape:
        imgs = reshape_imgs(imgs, size, mode)
    else:
        return imgs.permute(0, 3, 1, 2)

def create_data_loader(folder_path_list, batch_size, shuffle):

    inputs, labels = load_concat_data(folder_path_list)
    dataset = []
    for i in range(len(inputs)):
        dataset.append([inputs[i], labels[i]])   
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)