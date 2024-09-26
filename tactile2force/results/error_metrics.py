import numpy as np

def comute_avg_l2_norm(prediction, ground_truth):
    """
    Compute the average L2 norm between the prediction and the ground truth.
    Args:
        prediction: size (n, m) numpy array where n is the number of data points and m is the dimension of the data points.
        ground_truth: size (n, m) numpy array where n is the number of data points and m is the dimension of the data points.
    Returns:
        The average L2 norm between the prediction and the ground truth.
    """
    n = prediction.shape[0]
    m = prediction.shape[1]

    err = prediction - ground_truth
    err = np.square(err)
    err = np.sqrt(np.sum(err, axis=1))
    return np.sum(err) / n

def MSE(prediction, ground_truth):
    """
    Computes the mean square error in the same way pytorch MSELoss() does with its default parameters (reduce = True and reduction = 'mean').
    Args:
        prediction: size (n, m) numpy array where n is the number of data points and m is the dimension of the data points.
        ground_truth: size (n, m) numpy array where n is the number of data points and m is the dimension of the data points.
    Returns:
        The mean square error between the prediction and the ground truth.
    """
    n = prediction.shape[0]
    m = prediction.shape[1]

    err = prediction - ground_truth
    err = np.square(err)
    err = 1/m*np.sum(err, axis=1)
    return np.sum(err) / n

def MSE_norm(predicted_norm, true_norm):
    n = predicted_norm.shape[0]

    err = predicted_norm - true_norm
    return 1/n*np.sum(np.square(err))

def get_norm(f, scale):
     return np.sqrt(np.square(f[:,0]*scale[0])+np.square(f[:,1]*scale[1])+np.square(f[:,2]*scale[2]))

def get_rel_error(F_est, F, threshold = 0.5, get_std=False):
    F_est = F_est[abs(F)>threshold]
    F = F[abs(F)>threshold]

    if get_std:
        return np.mean(np.abs(np.abs(F_est - F) / F)), np.std(np.abs(np.abs(F_est - F) / F))

    return np.mean(np.abs(np.abs(F_est - F) / F))

def get_rel_error_inferior_to(F_est, F, min_threshold = 0.5, max_threshold = 3.0):
    F_est = F_est[(abs(F)>min_threshold) & (abs(F)<max_threshold)]
    F = F[(abs(F)>min_threshold) & (abs(F)<max_threshold)]

    return np.mean(np.abs(np.abs(F_est - F) / F))