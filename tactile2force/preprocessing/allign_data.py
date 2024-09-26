import pickle
import matplotlib.pyplot as plt
import os
import json
import preprocessing.preprocessing_functions_copy as pp
import preprocessing.graphical_functions as gf
import parameters.xela_params as params

def do_it(input_bag, output_pickle, patch_of_interest, verbose=False, cut_data=False, cut_time=0.0):
    '''
    This function processes the data from the input bag file and saves the preprocessed data to pickle files.
    args:
        input_bag (str): The path to the input bag file.
        output_pickle (str): The path to the output pickle files.
        patch_of_interest (str): The id of the patch of interest.
    returns:
        None
    '''
    format = None
    transform = None
    x_idx = []
    y_idx = []
    z_idx = []

    if patch_of_interest in params.tip_patches:
        format = 'Tip'
        x_idx = list(range(1, 31))
        y_idx = list(range(31, 61))
        z_idx = list(range(61, 91))
        if patch_of_interest in params.index_finger:
            transform = '/sensor_to_index'
        elif patch_of_interest in params.middle_finger:
            transform = '/sensor_to_middle'
        elif patch_of_interest in params.ring_finger:
            transform = '/sensor_to_ring'
        elif patch_of_interest in params.thumb_finger:
            transform = '/sensor_to_thumb'
    elif patch_of_interest in params.phalanges_patches:
        x_idx = list(range(1, 17))
        y_idx = list(range(17, 33))
        z_idx = list(range(33, 49))
        if patch_of_interest == '6A':
            transform = '/sensor_to_middle_3rd_phal'


    # extract from bag and csv
    csv_path_list, topics = pp.bag_to_csv(input_bag)
    df_dict = pp.csv_to_df(csv_path_list, topics , drop=['header.seq', 'header.frame_id', 'sensor', 'model'])

    # convert data and time to np arrays
    tactile_data, tactile_time = pp.xela_df_to_np(df_dict['/xServTopic'])
    patch_data = pp.get_tactile_patch(tactile_data, patch_of_interest)
    tf_data, tf_time = pp.process_tfs(df_dict[transform])
    force_data, force_time = pp.force_df_to_np(df_dict['/ft_sensor/netft_data'])

    # set reference time to zero
    tactile_time = pp.set_ref_time_to_zero(tactile_time)
    tf_time = pp.set_ref_time_to_zero(tf_time)
    force_time = pp.set_ref_time_to_zero(force_time)

    if cut_data:
        patch_data = patch_data[tactile_time[:,0] < cut_time]
        tf_data = tf_data[tf_time[:,0] < cut_time]
        force_data = force_data[force_time[:,0] < cut_time]
    
    # AUTOSYNC
    force_legends = ['Fx', 'Fy', 'Fz']
    tactile_legends = ['x', 'y', 'z']

    # INTERPOLATION
    if verbose: print("Interpolating signals...")
    tact_sync, force_sync, tf_sync = pp.interpolate_signals(patch_data, force_data, tf_data)

    if verbose: print("Rotating forces...")
    force_sync_rotated = pp.rotate_forces(force_sync, tf_sync)

    if verbose: print("Filtering forces...")
    filt_forces = force_sync_rotated
    # pp.ema_3d(force_sync_rotated, 0.15)

    if verbose: print("Normalizing...")

    t1, t2 = gf.select_range(filt_forces[:,1:], force_legends)
    deb_forces, force_meta = pp.debias_q95_norm(filt_forces, 1, 2, 3, t1, t2)

    tactile_viz = pp.tip_to_centered_avg_per_axis(tact_sync)
    t1, t2 = gf.select_range(tactile_viz[:,1:4], force_legends)
    deb_tact, tact_meta = pp.debias_q95_norm(tact_sync, x_idx, y_idx, z_idx, t1, t2)

    deb_forces = pp.filter_force(deb_forces)
    deb_tact = pp.filter_tactile(deb_tact, window=10, kernel='mavg')

    try:
        os.makedirs(output_pickle)
    except FileExistsError:
        print("Output directory already exists.")
    except PermissionError:
        print("Insufficient permissions to create the directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
    else:
        print("Output directory created successfully.")

    with open(output_pickle+'/'+'tact_data.pkl', 'wb') as f:
        pickle.dump(deb_tact[:,1:], f)

    with open(output_pickle+'/'+'force_label.pkl', 'wb') as f:
        pickle.dump(deb_forces[:,1:], f)
    
    metadata = {'Force':force_meta, 'Tactile':tact_meta}

    # Save dictionary to JSON file at a given file path
    file_path = output_pickle+'/metadata.json'
    with open(file_path, 'w') as f:
        json.dump(metadata, f)

    # Create figure and axes objects
    fig, axs = plt.subplots(2, sharex=True)

    # Plot Signal 1
    axs[0].plot(filt_forces[:,0], filt_forces[:,1:], label='Forces')
    axs[0].set_ylabel('Normalized Force')
    axs[0].set_title('Forces vs Time')
    axs[0].legend()

    # Plot Signal 2
    axs[1].plot(filt_forces[:,0], deb_tact[:,1:], label='Tactile')
    axs[1].set_ylabel('Normalized tactile')
    axs[1].set_title('Tactile vs Time')
    axs[1].legend()

    # Set common x-axis label for both plots
    fig.text(0.5, 0.04, 'Time (s)', ha='center')

    plt.show()