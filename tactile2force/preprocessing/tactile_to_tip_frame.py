import numpy as np

def rpy_to_rotation_matrix(rpy_array, sensor_o2fingertip, measurment2taxel):
    '''
    args:
        rpy_array: np.array of shape (n_taxels, 3)
        sensor_o2fingertip: np.array of shape (3, 3)
        measurment2taxel: np.array of shape (3, 3)
    returns:
        rotation_matrix: np.array of shape (n_taxels, 3, 3)
    '''
    n_taxels = rpy_array.shape[0]
    rotation_matrix = np.zeros((n_taxels, 3, 3))
    for i in range(n_taxels):
        roll_matrix = np.array([[1, 0, 0],
                                [0, np.cos(rpy_array[i, 0]), -np.sin(rpy_array[i, 0])],
                                [0, np.sin(rpy_array[i, 0]), np.cos(rpy_array[i, 0])]])
        
        pitch_matrix = np.array([[np.cos(rpy_array[i, 1]), 0, np.sin(rpy_array[i, 1])],
                                [0, 1, 0],
                                [-np.sin(rpy_array[i, 1]), 0, np.cos(rpy_array[i, 1])]])
        
        yaw_matrix = np.array([[np.cos(rpy_array[i, 2]), -np.sin(rpy_array[i, 2]), 0],
                                [np.sin(rpy_array[i, 2]), np.cos(rpy_array[i, 2]), 0],
                                [0, 0, 1]])
        
        rot = np.dot(yaw_matrix, np.dot(pitch_matrix, roll_matrix))
        rotation_matrix[i] = np.dot(sensor_o2fingertip, np.dot(rot, measurment2taxel))
    return rotation_matrix

def reshape_tactile_by_dim(tactile, n_taxels):
    x = tactile[:, 0:n_taxels]
    y = tactile[:, n_taxels:2*n_taxels]
    z = tactile[:, 2*n_taxels:]
    return np.stack([x, y, z], axis=2)

def rotate_tactile(tactile, rot_matrix):
    '''
    args:
        rot_matrix: np.array of shape (n_taxels, 3, 3) [i j k]
        tactile: np.array of shape (n_samples, n_taxels, 3) []
    returns:
        rotated_tactile: np.array of shape (n_samples, n_taxels, 3)
    '''
    #tactile = reshape_tactile_by_dim(tactile, tactile.shape[1]//3)
    rotated_tactile = np.einsum('...ij,...j->...i', rot_matrix, tactile)
    return rotated_tactile

def rotate_phal_patch(tactile, rot):
    '''
    args:
        tactile: np.array of shape (n_samples, n_taxels, 3)
        rot: np.array of shape (3, 3)
    returns:
        rotated_tactile: np.array of shape (n_samples, n_taxels, 3)
    '''
    return np.einsum('...ij,...j->...i', rot, tactile)