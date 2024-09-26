import numpy as np
from datetime import datetime
import models.m5_fully_connected as net
from results.error_metrics import MSE, MSE_norm, get_norm, get_rel_error, get_rel_error_inferior_to
from results.save_results import save_performance
import test_model as tm
import matplotlib.pyplot as plt

def run_m5(n_train, n_test, finger, patch_type):
    model_name = 'M5'
    model = net.NeuralNetwork(patch_type)

    train, val, x_train, x_val, standardization_dict = tm.prep_data_nn(n_train, n_test, finger, patch_type, 256, True, 5)
    std_tactile = standardization_dict["tactile_scale"]
    std_force = standardization_dict["force_scale"]

    train_loss, val_loss = net.train_network(model, x_train, x_val, num_epochs=80, learning_rate=0.00025)
    #train_loss, val_loss = net.train_network(model2, x_train, x_val, num_epochs=300, learning_rate=0.00025)
    train_fit = net.predict(model, x_train.dataset.data)
    #train_fit = net_tanh.predict(model2, x_train.dataset)
    test_fit = net.predict(model, x_val.dataset.data)

    F_train = get_norm(x_train.dataset.labels.numpy(), std_force)
    F_train_fit = get_norm(train_fit.detach().numpy(), std_force)
    F_val = get_norm(x_val.dataset.labels.numpy(), std_force)
    F_val_fit = get_norm(test_fit.detach().numpy(), std_force)

    train_MSE = MSE(train_fit.detach().numpy()*np.array(std_force), x_train.dataset.labels.numpy()*np.array(std_force))
    test_MSE = MSE(test_fit.detach().numpy()*np.array(std_force), x_val.dataset.labels.numpy()*np.array(std_force))
    train_norm_MSE = MSE_norm(F_train_fit, F_train)
    test_norm_MSE = MSE_norm(F_val_fit, F_val)

    print("Errors: ")
    print(test_norm_MSE)
    print(train_MSE)
    print(test_MSE)
    rel_err = get_rel_error(x_val.dataset.labels.numpy(), test_fit.detach().numpy())
    print('relative ertror is: ', rel_err.item())
    rel_err_inferior_to = get_rel_error_inferior_to(x_val.dataset.labels.numpy(), test_fit.detach().numpy())
    print('relative ertror is: ', rel_err_inferior_to.item())

    # Timestamp and save train and test metadata
    now = datetime.now()
    save_performance(now, model_name, train, val, train_MSE, test_MSE, train_norm_MSE, test_norm_MSE, rel_err.item(), rel_err_inferior_to.item(), standardization_dict, np.min(train_loss).item(), np.min(val_loss).item())
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")

    #print(model.model_params)

    tm.plot_and_save_results(model_name, finger, now_str, x_train.dataset.labels, train_fit.detach().numpy(), x_val.dataset.labels, test_fit.detach().numpy(), std_force, F_train_fit, F_train, F_val_fit, F_val)

if __name__ == '__main__':
    n_train = 4
    n_test = 2
    finger = 'index'
    patch_type = 'tip'
    run_m5(n_train, n_test, finger, patch_type)