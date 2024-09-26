import numpy as np

def get_polynomial_regressors_2(data):
    augmented = np.zeros((data.shape[0], 3*data.shape[1]))
    augmented[:, :data.shape[1]] = data
    augmented[:, data.shape[1]:2*data.shape[1]] = data[:,0:data.shape[1]] * data[:,0:data.shape[1]]
    augmented[:, 2*data.shape[1]: 2*data.shape[1] + 30] = data[:,0:30] * data[:,30:60]
    augmented[:, 2*data.shape[1] + 30: 2*data.shape[1] + 60] = data[:,0:30] * data[:,60:90]
    augmented[:, 2*data.shape[1] + 60:] = data[:,30:60] * data[:,60:90]
    return augmented

def get_polynomial_regressors(x):
    x_x = x[:, :, 0]
    x_y = x[:, :, 1]
    x_z = x[:, :, 2]
    x_xx = x_x*x_x
    x_yy = x_y*x_y
    x_zz = x_z*x_z
    x_xy = x_x*x_y
    x_xz = x_x*x_z
    x_yz = x_y*x_z

    return np.stack((x_x, x_y, x_z, x_xx, x_yy, x_zz, x_xy, x_xz, x_yz), axis=-1)

def get_polynomial_regressors_proj(data):
    x_x = data[:, 0]
    x_y = data[:, 1]
    x_z = data[:, 2]
    x_xx = x_x*x_x
    x_yy = x_y*x_y
    x_zz = x_z*x_z
    x_xy = x_x*x_y
    x_xz = x_x*x_z
    x_yz = x_y*x_z

    return np.stack((x_x, x_y, x_z, x_xx, x_yy, x_zz, x_xy, x_xz, x_yz), axis=-1)