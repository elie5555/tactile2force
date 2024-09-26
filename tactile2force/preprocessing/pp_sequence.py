import pickle
import matplotlib.pyplot as plt
import os
import json
import preprocessing.preprocessing_functions as pp
import preprocessing.graphical_functions as gf
import parameters.xela_params as params

def process_bag(input_bag):
    # extract from bag and csv
    csv_path_list, topics = pp.bag_to_csv(input_bag)
    df_dict = pp.csv_to_df(csv_path_list, topics , drop=['header.seq', 'header.frame_id', 'sensor', 'model'])
    return df_dict

def dict2np(df_dict, patch_of_interest):
    transform = params.patch2tf[patch_of_interest]

    # convert data and time to np arrays
    tactile_data, tactile_time = pp.xela_df_to_np(df_dict['/xServTopic'])
    patch_data = pp.get_tactile_patch(tactile_data, patch_of_interest)
    tf_data, tf_time = pp.process_tfs(df_dict[transform])
    force_data, force_time = pp.force_df_to_np(df_dict['/ft_sensor/netft_data'])

    # set reference time to zero
    tactile_time = pp.set_ref_time_to_zero(tactile_time)
    tf_time = pp.set_ref_time_to_zero(tf_time)
    force_time = pp.set_ref_time_to_zero(force_time)

    return patch_data, tf_data, force_data, tactile_time, tf_time, force_time

def do_it(input_bag, output_pickle, patch_of_interest, verbose=False, synchronize=False, cut_data=False, cut_time=0.0):
    '''
    This function processes the data from the input bag file and saves the preprocessed data to pickle files.
    args:
        input_bag (str): The path to the input bag file.
        output_pickle (str): The path to the output pickle files.
        patch_of_interest (str): The id of the patch of interest.
    returns:
        None
    '''
    df_dict = process_bag(input_bag)
    patch_data, tf_data, force_data, tactile_time, tf_time, force_time = dict2np(df_dict, patch_of_interest)

    if cut_data:
        patch_data = patch_data[tactile_time < cut_time]
        tactile_time = tactile_time[tactile_time < cut_time]

        tf_data = tf_data[tf_time < cut_time]
        tf_time = tf_time[tf_time < cut_time]

        force_data = force_data[force_time < cut_time]
        force_time = force_time[force_time < cut_time]
    
    # AUTOSYNC
    force_legends = ['Fx', 'Fy', 'Fz']
    tactile_legends = ['x', 'y', 'z']

    if synchronize:
        if verbose: print("Synchronizing signals... Click at the begginig of the signals and close plots once done.")
        tau_force = gf.plot_signal_with_click(force_time, force_data[:,0:3], force_legends)

        tactile_viz = pp.tip_to_centered_avg_per_axis(patch_data)
        tau_tact = gf.plot_signal_with_click(tactile_time, tactile_viz[:,0:3], tactile_legends)

        patch_data_alligned, tf_data_alligned, force_data_alligned = pp.allign_signals(patch_data, tf_data, force_data, tau_tact, tau_force, tau_force)
    else:
        patch_data_alligned = patch_data
        tf_data_alligned = tf_data
        force_data_alligned = force_data

    # INTERPOLATION
    if verbose: print("Interpolating signals...")
    time, tact_sync, force_sync, tf_sync = pp.interpolate_signals(patch_data_alligned, force_data_alligned, tf_data_alligned, tactile_time, force_time, tf_time)

    if verbose: print("Rotating forces...")
    force_sync_rotated = pp.rotate_forces(force_sync, tf_sync)

    if verbose: print("Filtering forces...")
    filt_forces = force_sync_rotated

    if verbose: print("Normalizing...")

    t1, t2 = gf.select_range(filt_forces, force_legends)
    deb_forces, force_meta = pp.debias_force(filt_forces, t1, t2)

    tactile_viz = pp.tip_to_centered_avg_per_axis(tact_sync)
    t1, t2 = gf.select_range(tactile_viz[:,0:3], force_legends)
    deb_tact, tact_meta = pp.debias_tactile(tact_sync, t1, t2)

    #deb_forces = pp.filter_force(time, deb_forces, cutoff=4, order=1)
    #deb_tact = pp.filter_tactile2(time, deb_tact, cutoff=4, order=1)

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
        pickle.dump(deb_tact, f)

    with open(output_pickle+'/'+'force_label.pkl', 'wb') as f:
        pickle.dump(deb_forces, f)

    with open(output_pickle+'/'+'time.pkl', 'wb') as f:
        pickle.dump(time, f)
    
    metadata = {'Force':force_meta, 'Tactile':tact_meta}

    # Save dictionary to JSON file at a given file path
    file_path = output_pickle+'/metadata.json'
    with open(file_path, 'w') as f:
        json.dump(metadata, f)

    # Create figure and axes objects
    fig, axs = plt.subplots(2, sharex=True)

    # Plot Signal 1
    axs[0].plot(time, filt_forces[:,0:], label='Forces')
    axs[0].set_ylabel('Normalized Force')
    axs[0].set_title('Forces vs Time')
    axs[0].legend()

    # Plot Signal 2
    axs[1].plot(time, deb_tact.reshape((deb_tact.shape[0], -1)), label='Tactile')
    axs[1].set_ylabel('Normalized tactile')
    axs[1].set_title('Tactile vs Time')
    axs[1].legend()

    # Set common x-axis label for both plots
    fig.text(0.5, 0.04, 'Time (s)', ha='center')

    plt.show()