import numpy as np
from datetime import datetime

from models.m4_parameter_free_polynomial import M4
from results.error_metrics import MSE, MSE_norm, get_norm, get_rel_error, get_rel_error_inferior_to
from results.save_results import save_performance
import test_model as tm

def run_m4(n_train, n_test, finger, patch_type):
    model = M4()
    model_name = 'M4'

    train, val, x_train, y_train, x_val, y_val, standardization_dict = tm.prep_data(n_train, n_test, finger, patch_type)
    std_tactile = standardization_dict["tactile_scale"]
    std_force = standardization_dict["force_scale"]

    train_fit = model.fit(x_train, y_train)
    test_fit = model.predict(x_val)

    F_train = get_norm(y_train, std_force)
    F_train_fit = get_norm(train_fit, std_force)
    F_val = get_norm(y_val, std_force)
    F_val_fit = get_norm(test_fit, std_force)

    train_MSE = MSE(train_fit*np.array(std_force), y_train*np.array(std_force))
    test_MSE = MSE(test_fit*std_force, y_val*std_force)
    train_norm_MSE = MSE_norm(F_train_fit, F_train)
    test_norm_MSE = MSE_norm(F_val_fit, F_val)
    print("Errors: ")
    print(test_norm_MSE)
    print(train_MSE)
    print(test_MSE)
    rel_err = get_rel_error(y_val, test_fit)
    print('relative error: ', rel_err*100, '%')
    rel_err_inferior_to = get_rel_error_inferior_to(y_val, test_fit)
    print('relative error: ', rel_err_inferior_to*100, '%')

    # Timestamp and save train and test metadata
    now = datetime.now()
    save_performance(now, model_name, train, val, train_MSE, test_MSE, train_norm_MSE, test_norm_MSE, rel_err, rel_err_inferior_to, standardization_dict)
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")

    #print(model.model_params)

    tm.plot_and_save_results(model_name, finger, now_str, y_train, train_fit, y_val, test_fit, std_force, F_train_fit, F_train, F_val_fit, F_val)

if __name__ == '__main__':
    n_train = 4
    n_test = 2
    finger = 'index'
    patch_type = 'tip'
    run_m4(n_train, n_test, finger, patch_type)