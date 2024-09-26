from dataset.load_data import get_tip_datapath, load_concat_data, get_phal_datapath, create_data_loader
from dataset.mydataset import MyCustomDataset
import matplotlib.pyplot as plt
import numpy as np

def prep_data(n_train, n_test, finger, patch_type):
    if patch_type == 'tip':
        train, val = get_tip_datapath(n_train, n_test, finger)
    elif patch_type == 'phal':
        train, val = get_phal_datapath(n_train, n_test, finger)

    x_train, y_train = load_concat_data(train)
    x_val, y_val = load_concat_data(val)

    train_dataset = MyCustomDataset(x_train, y_train, 'tab')
    standardization_dict = train_dataset.standardize_data()

    x_train = train_dataset.data.numpy()
    y_train = train_dataset.labels.numpy()

    std_tactile = standardization_dict["tactile_scale"]
    std_force = standardization_dict["force_scale"]

    x_val[:, :, 0] = x_val[:, :, 0]/std_tactile[0]
    x_val[:, :, 1] = x_val[:, :, 1]/std_tactile[1]
    x_val[:, :, 2] = x_val[:, :, 2]/std_tactile[2]

    y_val[:,0] = y_val[:,0]/std_force[0]
    y_val[:,1] = y_val[:,1]/std_force[1]
    y_val[:,2] = y_val[:,2]/std_force[2]

    x_val = x_val.numpy()
    y_val = y_val.numpy()

    return train, val, x_train, y_train, x_val, y_val, standardization_dict

def plot_and_save_results(model_name, finger, now_str, y_train, train_fit, y_val,
                           test_fit, std_force, F_train_fit, F_train, F_val_fit, F_val, autoclose=True):
    titles = ["X axis","Y axis", "Z axis"]

    # Plot the results
    # Create a figure and three subplots
    fig, axes = plt.subplots(4, 1, figsize=(8, 12))  # 3 rows, 1 column of subplots
    # Plot data on the subplots
    for i in [0,1,2]:
        axes[i].plot(y_train[:,i]*std_force[i], color='blue', label='true force')
        axes[i].plot(train_fit[:,i]*std_force[i], color='orange', label='regressed force')
        axes[i].set_title(titles[i])
        axes[i].legend()
        axes[i].grid()
        axes[i].set_xlabel('Sample')
        axes[i].set_ylabel('Force value [N]')
    axes[3].plot(F_train, color='blue', label='true norm')
    axes[3].plot(F_train_fit, color='orange', label='regressed norm')
    axes[3].set_title('Norm')
    axes[3].legend()
    axes[3].grid()
    axes[3].set_xlabel('Sample')
    axes[3].set_ylabel('Force value [N]')

    # Add a title to the whole figure
    fig.suptitle('Performances on the training data')
    # Add labels and adjust layout
    plt.tight_layout()
    # Save the plot
    plt.savefig(f'results/{model_name}/{now_str}-{finger}-train.pdf', format='pdf')
    # Show the plot
    plt.show(block=False)

    if autoclose:
        plt.close()

    # Plot the results
    # Create a figure and three subplots
    fig, axes = plt.subplots(4, 1, figsize=(8, 12))  # 3 rows, 1 column of subplots
    # Plot data on the subplots
    for i in [0,1,2]:
        axes[i].plot(y_val[:,i]*std_force[i], color='blue', label='true force')
        axes[i].plot(test_fit[:,i]*std_force[i], color='orange', label='regressed force')
        axes[i].set_title(titles[i])
        axes[i].legend()
        axes[i].grid()
        axes[i].set_xlabel('Sample')
        axes[i].set_ylabel('Force value [AU]')
        axes[3].plot(F_val, color='blue', label='true norm')
    axes[3].plot(F_val_fit, color='orange', label='regressed norm')
    axes[3].set_title('Norm')
    axes[3].legend()
    axes[3].grid()
    axes[3].set_xlabel('Sample')
    axes[3].set_ylabel('Force value [N]')

    # Add a title to the whole figure
    fig.suptitle('Performances on the validation data')
    # Add labels and adjust layout
    plt.tight_layout()
    # Save the plot
    plt.savefig(f'results/{model_name}/{now_str}-{finger}-val.pdf', format='pdf')
    # Show the plot
    plt.show(block=False)

    if autoclose:
        plt.close()


def prep_data_nn(n_train, n_test, finger, patch_type, batch_size, avg_pool, size):
    if patch_type == 'tip':
        train, val = get_tip_datapath(n_train, n_test, finger)
    elif patch_type == 'phal':
        train, val = get_phal_datapath(n_train, n_test, finger)
    # Load the data from datapaths generated above
    x_train, param_dict = create_data_loader(train, batch_size, True, regularize=True, avg_pool=avg_pool, avg_pool_size=size)
    x_val = create_data_loader(val, batch_size, True)

    # Standardize the validation data using the parameters from the training data
    #x_val.dataset.standardize_data_ext(param_dict)
    x_val.dataset.apply_standardization(param_dict)

    # Reshape the data to be in the correct format for the neural network
    x_train.dataset.reshape_to_vect()
    x_val.dataset.reshape_to_vect()
    
    return train, val, x_train, x_val, param_dict

def prep_data_cnn(n_train, n_test, finger, patch_type, batch_size, avg_pool, size):
    # Load the data from generated datapaths 
    if patch_type == 'tip':
        train, val = get_tip_datapath(n_train, n_test, finger)
        x_train, param_dict = create_data_loader(train, batch_size, True, regularize=True, as_img=True, avg_pool=avg_pool, avg_pool_size=size)
        x_val = create_data_loader(val, batch_size, True, as_img=True)
    elif patch_type == 'phal':
        train, val = get_phal_datapath(n_train, n_test, finger)
        x_train, param_dict = create_data_loader(train, batch_size, True, regularize=True, as_img=True, avg_pool=avg_pool, avg_pool_size=size)
        x_val = create_data_loader(val, batch_size, True, as_img=True)

    # Standardize the validation data using the parameters from the training data
    #x_val.dataset.standardize_data_ext(param_dict)
    x_val.dataset.apply_standardization(param_dict)
    
    return train, val, x_train, x_val, param_dict