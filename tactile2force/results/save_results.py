import json
from datetime import datetime
from results.error_metrics import get_rel_error

def model_data2dict(now, train_list, val_list, train_MSE, test_MSE, train_norm_MSE, test_norm_MSE, rel_err, rel_err_inf, scale_dict, train_LOSS, val_LOSS):
    '''
    This function converts the model data into a dictionary.
    Args:
        train_list: list, the list of training data.
        val_list: list, the list of validation data.
        train_MSE: training MSE value.
        test_MSE: test MSE value.
    Returns:
        model_data: dict, the model data.
    '''

    model_data = {}
    model_data['date'] = now.strftime("%Y-%m-%d-%H-%M-%S")
    model_data['train_set'] = train_list
    model_data['validation_set'] = val_list
    model_data['train_loss'] = train_LOSS
    model_data['validation_loss'] = val_LOSS
    model_data['train_MSE'] = train_MSE
    model_data['validation_MSE'] = test_MSE
    model_data['train_norm_MSE'] = train_norm_MSE
    model_data['validation_norm_MSE'] = test_norm_MSE
    model_data['relative_error'] = rel_err
    model_data['relative_error_small_forces'] = rel_err_inf
    model_data['force_scaling'] = scale_dict["force_scale"]
    model_data['tactile_scaling'] = scale_dict["tactile_scale"]
    return model_data

def save_performance(now, model_name, train_list, val_list, train_MSE, test_MSE, train_norm_MSE, test_norm_MSE, rel_err, rel_err_inf, scale_dict, train_LOSS=None, val_LOSS=None):
    '''
    This function saves the performance of the model.
    Args:
        model: str, the name of the model.
        train_list: list, the list of training data.
        val_list: list, the list of validation data.
        train_MSE: training MSE value.
        test_MSE: test MSE value.
    '''
    if not(model_name in ["M0", "M1", "M2", "M2NB", "M3", "M3nd", "M4", "M4x","M5", "M5L", "M6", "M7", "M8", "M9"]):
        raise ValueError("Model name not recognized")
    else:
        new_data = model_data2dict(now, train_list, val_list, train_MSE, test_MSE, train_norm_MSE, test_norm_MSE, rel_err, rel_err_inf, scale_dict, train_LOSS, val_LOSS)
        with open(f'results/{model_name}/results.jsonl', 'a') as f:
            f.write(json.dumps(new_data) + '\n')

