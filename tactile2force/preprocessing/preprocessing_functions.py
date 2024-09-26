import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate
from bagpy import bagreader
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm
import re
import os
import math
import scipy.stats as stats
import parameters.xela_params as xela_params

def format_topic_name(name):
    '''
    This function replaces '/' in a string either by '-' or removes it if it is the first character

    Arg:
        name (string): string to be formatted

    Returns:
        string: formatted string
    '''

    x = [i for i, ltr in enumerate(name) if ltr == '/']
    list_name = list(name)
    remove_first = False
    for id in x:
        if id:
            list_name[id] = '-'
        else:
            remove_first = True
    if remove_first:
        list_name.pop(0)
    return "".join(list_name)

def bag_to_csv(bag_file_path):
    '''
    This function reads a bag file and extracts all topics into csv files
    
    Args:
        bag_file_path (string): path to the bag file
        
    Returns:
        csv_path_list (list[string]): list of paths to the csv files
        topics (list[string]): list of topics
    '''
    bag = bagreader(bag_file_path)
    topics = list(bag.topic_table["Topics"])
    csv_path_list = []
    for top in tqdm(topics):
        path = bag_file_path.split('.')[0]
        path = path + '/' + format_topic_name(top) + ".csv"
        csv_path_list.append(path)
        if os.path.exists(path):
            print("Csv already exisits for {}, not creating".format(top))
            pass
        else:
            bag.message_by_topic(top)

    return csv_path_list, topics

def csv_to_df(files_name_list, topics, drop=[]):
    '''
    This function reads a list of csv files and returns a dictionary of dataframes

    Args:
        files_name_list (list[string]): list of paths to the csv files
        topics (list[string]): list of topics
        drop (list[string]): list of columns to drop from every dataframe if they exist
    
    Returns:
        df_dict (dict): dictionary of dataframes
    '''
    df_dict = {}
    for top, csv_file in tqdm(zip(topics, files_name_list)):
        df_dict[top] = pd.read_csv(csv_file)
        to_drop = list(set(drop) & set(df_dict[top].columns.tolist()))
        if len(to_drop):
            df_dict[top].drop(columns=to_drop, inplace=True)

    return df_dict

def xela_df_to_np(xela_df):
    '''
    This function processes the raw uskin data into a numpy array of size (samples x taxel) and removes instantaneous 0 drops (artifact)

    Args:
        xela_df (pandas.DataFrame): dataframe containing the raw data

    Returns:
        tactile_np (numpy.ndarray): numpy array of size (samples x taxel x 3)
        tactile_time (numpy.ndarray): numpy array of size (samples x 1) containing time 
    '''

    n_samples = xela_df.shape[0]

    array_shape = (n_samples, 368, 3)
    tactile_np = np.zeros(array_shape)

    i = 0
    for raw_data_point in tqdm(xela_df['points']):
        # Extract numbers using regular expressions
        tactile_np[i, :, 0] = re.findall(r'x: (\d+\.\d+)', raw_data_point)
        tactile_np[i, :, 1] = re.findall(r'y: (\d+\.\d+)', raw_data_point)
        tactile_np[i, :, 2] = re.findall(r'z: (\d+\.\d+)', raw_data_point)
        i = i+1
    
    #get rid of 0 drop artefacts in the signal by replacing it by closest past non 0 value
    if np.any(tactile_np[0,:,:] == 0):
        idx = np.where(tactile_np[0,:,:] == 0)
        #print(idx[0])
        #print(idx[1])
        for j, k in zip(idx[0], idx[1]):
            i = 1
            while (tactile_np[i,j,k] == 0):
                i = i+1
                if (i == tactile_np.shape[0] - 1):
                    break
            tactile_np[0,j,k] = tactile_np[i,j,k]

    prev_row = tactile_np[0,:,:]
    for i in tqdm(range(1, tactile_np.shape[0])):
        row = tactile_np[i,:,:]
        temp = row.astype(bool)
        temp = ~temp
        tactile_np[i,:,:] = row + temp.astype(int) * prev_row
        prev_row = tactile_np[i,:,:]

    #recompute time col using s and ns
    tactile_time = xela_df['header.stamp.secs'].to_numpy() + xela_df['header.stamp.nsecs'].to_numpy()*1e-9

    return tactile_np, tactile_time

def extract_transform(in_str):
    '''
    This function extracts the transform (ROS tf) from a string

    Args:
        in_str (string): string containing the transform

    Returns:
        nums (list[float]): list of floats containing the timestamp and the transform [t, x, y, z, qx, qy, qz, qw]
    '''

    x = in_str[1:-1].split("\n")
    keep_idx = [3, 4, 9, 10, 11, 13, 14, 15, 16]
    nums = [float(x[i].split(":")[1]) for i in keep_idx]
    return nums

def process_tfs(tf_dataframe):
    cols = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    clean_df = pd.DataFrame(columns=cols)
    for col in cols:
        clean_df[col]=clean_df[col].astype(float)
    for idx, row in tqdm(tf_dataframe.iterrows()):
        nums = extract_transform(row['transforms'])
        clean_df.at[idx, 't'] = nums[0] + nums[1]*1e-9
        clean_df.at[idx, 'x'] = nums[2]
        clean_df.at[idx, 'y'] = nums[3]
        clean_df.at[idx, 'z'] = nums[4]
        clean_df.at[idx, 'qx'] = nums[5]
        clean_df.at[idx, 'qy'] = nums[6]
        clean_df.at[idx, 'qz'] = nums[7]
        clean_df.at[idx, 'qw'] = nums[8]  

    tfs = clean_df.to_numpy()
    return tfs[:,1:], tfs[:,0]

def get_tactile_patch(tactile, patch):
    return tactile[:,xela_params.patch[patch],:]

def force_df_to_np(force_df):
    '''
    This function converts a force dataframe to a numpy array

    Args:
        force_df (pandas.DataFrame): dataframe containing the force data

    Returns:
        force_np (numpy.ndarray): numpy array of size (samples x 7) containing the force/torque data
    '''

    time = force_df['header.stamp.secs'].to_numpy() + force_df['header.stamp.nsecs'].to_numpy()*1e-9
    force = force_df[['wrench.force.x', 'wrench.force.y', 'wrench.force.z', 'wrench.torque.x', 'wrench.torque.y', 'wrench.torque.z']].to_numpy()
    return force, time

def viz_data(time, data, xlabel, ylabel, start_time=0, legend=[]):
    '''
    This function plots the data

    Args:
        time (numpy.ndarray): time data [samples x 1]
        data (numpy.ndarray): data to plot [samples x dim]
        xlabel (string): x axis label
        ylabel (string): y axis label
        start_time (float): time at which the horizontal line should be drawn (for signal synchronization)
        legend (list[string]): list of strings to use as legend

    Returns:
        None
    '''

    for i in range (data.shape[1]):
        if len(legend):
            plt.plot(time, data[:,i], label=legend[i])
        else:
            plt.plot(time, data[:,i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if start_time:
        plt.axvline(x=start_time, color='r', linestyle='--')
    if len(legend):
        plt.legend()
    plt.show()

def tip_to_centered_avg_per_axis(patch_data):
    '''
    This function computes the average of the taxels per axis and centers the data around 0. It is specific
    to the a fingertip patch of the Xela sensor

    args:
        patch_data (numpy.ndarray): data from the tip patch [samples x 91]
        patch_id (string): the type of patch 

    returns:
        centered_data (numpy.ndarray): centered data [samples x (x, y, z) = 4]
    '''
    patch_data = patch_data.mean(axis=1)
    return patch_data - patch_data[0:200,:].mean(axis=0)

def set_ref_time_to_zero(t):
    '''
    This function sets the reference time to 0

    Args:
        t (numpy.ndarray): data [n_], with time in the first column

    Returns:
        x (numpy.ndarray): data with time starting at 0
    '''

    t = t - t[0]
    return t

def remove_neg_time(x, t):
    '''
    This function removes the rows with negative time

    Args:
        x (numpy.ndarray): data [samples x dim], with time in the first column
    
    Returns:
        x (numpy.ndarray): data with no negative time
    '''

    return x[t >= 0]

def allign_signals(x, y, z, tau_x, tau_y, tau_z, tx, ty, tz):
    tau_min = min(tau_x, tau_y, tau_z)
    tau_x = tau_x - tau_min
    tau_y = tau_y - tau_min
    tau_z = tau_z - tau_min

    tx = tx - tau_x
    ty = ty - tau_y
    tz = tz - tau_z

    x = remove_neg_time(x, tz)
    y = remove_neg_time(y, ty)
    z = remove_neg_time(z, tz)

    return x, y, z

def interpolate_signals(x, y, z, tx, ty, tz):
    '''
    This function interpolates the signals to have the same time base

    Args:
        x (numpy.ndarray): data [samples x 3*n_taxel], with time in the first column
        y (numpy.ndarray): data [samples x 3], with time in the first column (forces)
        z (numpy.ndarray): data [samples x 4], with time in the first column (orientation quaternion)

    Returns:
        x (numpy.ndarray): data [samples x dim], with time in the first column
        y (numpy.ndarray): data [samples x dim], with time in the first column
        z (numpy.ndarray): data [samples x dim], with time in the first column
    '''

    # all signals will be interpolated using the time of the first signal
    # y is expected to be 3D
    # z is expected to be 4D

    x = x[tx <= min(ty[-1], tz[-1]), :, :]
    tx = tx[tx <= min(ty[-1], tz[-1])]
    x = x[tx >= max(ty[0], tz[0]), :, :]
    tx = tx[tx >= max(ty[0], tz[0])]

    y_1 = scipy.interpolate.interp1d(ty, y[:,0])
    y_2 = scipy.interpolate.interp1d(ty, y[:,1])
    y_3 = scipy.interpolate.interp1d(ty, y[:,2])

    y_1_interpolated = y_1(tx)
    y_2_interpolated = y_2(tx)
    y_3_interpolated = y_3(tx)

    z_1 = scipy.interpolate.interp1d(tz, z[:,3])
    z_2 = scipy.interpolate.interp1d(tz, z[:,4])
    z_3 = scipy.interpolate.interp1d(tz, z[:,5])
    z_4 = scipy.interpolate.interp1d(tz, z[:,6])

    z_1_interpolated = z_1(tx)
    z_2_interpolated = z_2(tx)
    z_3_interpolated = z_3(tx)
    z_4_interpolated = z_4(tx)

    y_sync = np.column_stack((y_1_interpolated, y_2_interpolated, y_3_interpolated))
    z_sync = np.column_stack((z_1_interpolated, z_2_interpolated, z_3_interpolated, z_4_interpolated))

    return tx, x, y_sync, z_sync

def rotate_forces(f, tf):
    '''
    This function rotates the forces using the orientation quaternion

    Args:
        f (numpy.ndarray): force data [samples x 4], with time in the first column
        tf (numpy.ndarray): transform data [samples x 8], with time in the first column

    Returns:
        f_tf (numpy.ndarray): rotated force data [samples x 4], with time in the first column
    '''

    # Convert quaternions to rotation matrices
    q = Rot.from_quat(tf)
    R_mat = q.as_matrix()
    # quaternions represent the rotation from tip to sensor so the rotation matrix needs to be inverted OR transposed
    R_mat = np.transpose(R_mat, axes=(0, 2, 1))
    f_tf = np.einsum('ijk,ik->ij', R_mat, f)

    return -f_tf

def ema(signal, alpha):
    '''
    This function computes the exponential moving average of a signal

    Args:
        signal (numpy.ndarray): signal to smooth
        alpha (float): smoothing factor

    Returns:
        out (numpy.ndarray): smoothed signal
    '''

    out = np.zeros(signal.shape)
    out[0] = signal[0]
    for i in range(0, signal.shape[0]):
        out[i] = alpha * signal[i] + (1-alpha) * out[i-1]
    return out

def ema_3d(data, alpha):
    '''
    This function computes the exponential moving average of a 3D signal

    Args:
        data (numpy.ndarray): data to smooth [samples x 4] (first col is time)
        alpha (float): smoothing factor

    Returns:
        data_ema (numpy.ndarray): smoothed data
    '''

    data_ema = np.empty_like(data)
    data_ema[:, 0] = ema(data[:, 0], alpha)
    data_ema[:, 1] = ema(data[:, 1], alpha)
    data_ema[:, 2] = ema(data[:, 2], alpha)
    return data_ema

def debias_q95_tactile(x, tau1, tau2):
    bias_x = x[tau1:tau2, :, 0].mean(axis=0)
    x[:,:,0] = x[:,:,0] - bias_x
    bias_y = x[tau1:tau2, :, 1].mean(axis=0)
    x[:,:,1] = x[:,:,1] - bias_y
    bias_z = x[tau1:tau2, :, 2].mean(axis=0)
    x[:,:,2] = x[:,:,2] - bias_z

    bias_x = bias_x.tolist()
    bias_y = bias_y.tolist()
    bias_z = bias_z.tolist()

    bias = [bias_x, bias_y, bias_z]

    qx = np.percentile(np.abs(x[:,:,0]), 99, axis=0)
    qy = np.percentile(np.abs(x[:,:,1]), 99, axis=0)
    qz = np.percentile(np.abs(x[:,:,2]), 99, axis=0)

    qx = np.mean(qx)
    qy = np.mean(qy)
    qz = np.mean(qz)

    x[:,:,0] = x[:,:,0] / qx
    x[:,:,1] = x[:,:,1] / qy
    x[:,:,2] = x[:,:,2] / qz

    metadata={'bias':bias, 'x_scale':qx, 'y_scale':qy, 'z_scale':qz}

    return x, metadata

def debias_force(x, tau1, tau2):
    avg = x[tau1:tau2].mean(axis=0)
    x = x - avg
    avg = avg.tolist()
    metadata={'avg':avg}
    return x, metadata

def debias_tactile(x, tau1, tau2):
    bias_x = x[tau1:tau2, :, 0].mean(axis=0)
    x[:,:,0] = x[:,:,0] - bias_x
    bias_y = x[tau1:tau2, :, 1].mean(axis=0)
    x[:,:,1] = x[:,:,1] - bias_y
    bias_z = x[tau1:tau2, :, 2].mean(axis=0)
    x[:,:,2] = x[:,:,2] - bias_z

    bias_x = bias_x.tolist()
    bias_y = bias_y.tolist()
    bias_z = bias_z.tolist()

    bias = [bias_x, bias_y, bias_z]
    metadata={'avg':bias}

    return x, metadata

def debias_q95_force(x, tau1, tau2):
    '''
    This function debiases the signal by removing the mean of the first tau samples and normalizes the signal

    Args:
        x (numpy.ndarray): data to process [samples x dim]
        x_idx (list[int]): indices of the x values in the data
        y_idx (list[int]): indices of the y values in the data
        z_idx (list[int]): indices of the z values in the data
        tau (int): number of samples to use for the mean computation

    Returns:
        x (numpy.ndarray): processed data
    '''

    avg = x[tau1:tau2, 0:].mean(axis=0)
    x[:,0:] = x[:,0:] - avg
    avg = avg.tolist()

    x_sig = x[:, 0]
    y_sig = x[:, 1]
    z_sig = x[:, 2]

    qx = np.percentile(np.abs(x_sig), 99, axis=0)
    qy = np.percentile(np.abs(y_sig), 99, axis=0)
    qz = np.percentile(np.abs(z_sig), 99, axis=0)

    qx = np.mean(qx)
    qy = np.mean(qy)
    qz = np.mean(qz)

    x[:, 0] = x_sig / qx
    x[:, 1] = y_sig / qy
    x[:, 2] = z_sig / qz

    metadata={'avg':avg, 'x_scale':qx, 'y_scale':qy, 'z_scale':qz}

    return x, metadata

def filter_force(time, force, cutoff=4.0, order=1):
    '''
    This function filters the force signal using a low-pass filter
    args:
        force (numpy.ndarray): force data [samples x 4], with time in the first column
        cutoff (float): cutoff frequency

    returns:
        filtered_force (numpy.ndarray): filtered force data [samples x 4], with time in the first column
    '''
    filtered_force = force.copy()
    dt = np.mean(time[1:] - time[:-1])
    #print("dt is: ", dt)
    fs = 1.0/dt

    # Design a Butterworth filter with cutoff frequency of 10 Hz and order of 4
    #order = 16  # filter order
    b, a = scipy.signal.butter(order, cutoff / (fs / 2), btype='lowpass', analog=False)

    # Create a figure with three subplots
    #fig, axs = plt.subplots(3, figsize=(10, 8))
    #labels=['X force', 'Y force', 'Z force']

    for i in [0,1,2]:

        # Apply the filter to the signal using filtfilt
        filtered_force[:,i] = scipy.signal.filtfilt(b, a, force[:,i])

        # Plot the original and filtered signals
        #axs[i].plot(time, force[:,i], label='Original')
        #axs[i].plot(time, filtered_force[:,i], label='Filtered')
        #axs[i].set_xlabel('Time [s]')
        #axs[i].set_ylabel(labels[i])
        #axs[i].legend()
    
    #plt.subplots_adjust(hspace=0.5)
    #plt.show()
    return filtered_force

def filter_tactile2(time, tactile, cutoff=4.0, order=1):
    '''
    This function filters the force signal using a low-pass filter
    args:
        force (numpy.ndarray): force data [samples x 4], with time in the first column
        cutoff (float): cutoff frequency

    returns:
        filtered_force (numpy.ndarray): filtered force data [samples x 4], with time in the first column
    '''
    filtered_tactile = tactile.copy()
    dt = np.mean(time[1:] - time[:-1])
    #print("dt is: ", dt)
    fs = 1.0/dt

    # Design a Butterworth filter with cutoff frequency of 10 Hz and order of 4
    #order = 16  # filter order
    b, a = scipy.signal.butter(order, cutoff / (fs / 2), btype='lowpass', analog=False)

    # Create a figure with three subplots
    #fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))

    #ints = np.random.choice(np.arange(0, tactile.reshape(tactile.shape[0], -1, 3).shape[1]), size=6, replace=False)
    #j = 0
    #k = 0
    for i in range(0, tactile.shape[1]):
        filtered_tactile[:,i, 0] = scipy.signal.filtfilt(b, a, tactile[:,i, 0])
        filtered_tactile[:,i, 1] = scipy.signal.filtfilt(b, a, tactile[:,i, 1])
        filtered_tactile[:,i, 2] = scipy.signal.filtfilt(b, a, tactile[:,i, 2])
        #if i in ints:
            #axs[j//2, (j+1)%2].plot(time, tactile.reshape((tactile.shape[0], -1, 3))[:,i, k], label='Original')
            #axs[j//2, (j+1)%2].plot(time, filtered_tactile.reshape((tactile.shape[0], -1, 3))[:,i, k], label='Filtered')
            #axs[j//2, (j+1)%2].set_xlabel('Time [s]')
            #axs[j//2, (j+1)%2].set_ylabel(f'Taxel {i}')
            #axs[j//2, (j+1)%2].legend()
            #j+=1
            #k = k+1
            #if k == 2:
                #k = 0

    #plt.subplots_adjust(hspace=0.5, wspace=0.5)
    #plt.show()
    return filtered_tactile.reshape((tactile.shape[0], -1, 3))

def filter_tactile(time, tactile, window=10, kernel='mavg'):
    '''
    This function filters the tactile signals using a moving avg filter w/ lag compensation
    args:
        tactile (numpy.ndarray): force data [samples x 91], with time in the first column
        cutoff (float): cutoff frequency

    returns:
        filtered_tactile (numpy.ndarray): filtered force data [samples x 4], with time in the first column
    '''
    tactile = tactile.reshape((tactile.shape[0], -1))
    filtered_tactile = tactile.copy()
    #print("filtered tactile shape: ", filtered_tactile.shape)
    dt = np.mean(time[1:] - time[:-1])
    fs = 1.0/dt

    if kernel == 'mavg':
        # Design a Butterworth filter with cutoff frequency of 10 Hz and order of 4
        kernel_val = np.ones(window) / window
    elif kernel == 'gaussian':
        mu = 0
        variance = 1
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, window)
        gaussian_kernel = stats.norm.pdf(x, mu, sigma)
        kernel_val = gaussian_kernel / np.sum(gaussian_kernel)

    # Create a figure with three subplots
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))

    ints = np.random.choice(np.arange(0, tactile.reshape(tactile.shape[0], -1, 3).shape[1]), size=6, replace=False)
    j = 0
    k = 0
    for i in range(1, tactile.shape[1]):
        filtered_tactile[:,i] = np.convolve(tactile[:,i], kernel_val, mode='same')
        if i in ints:
            axs[j//2, (j+1)%2].plot(time, tactile.reshape((tactile.shape[0], -1, 3))[:,i, k], label='Original')
            axs[j//2, (j+1)%2].plot(time, filtered_tactile.reshape((tactile.shape[0], -1, 3))[:,i, k], label='Filtered')
            axs[j//2, (j+1)%2].set_xlabel('Time [s]')
            axs[j//2, (j+1)%2].set_ylabel(f'Taxel {i}')
            axs[j//2, (j+1)%2].legend()
            j+=1
            k = k+1
            if k == 2:
                k = 0

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()
    return filtered_tactile.reshape((tactile.shape[0], -1, 3))