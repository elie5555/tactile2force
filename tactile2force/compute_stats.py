import json
import numpy as np

path = 'results/'
folders = ['M0/', 'M1/', 'M2/', 'M2NB/', 'M3/', 'M3nd/', 'M4/', 'M4x/', 'M5/', 'M5L/','M6/', 'M7/', 'M8/']
file = 'results.jsonl'

index_train_MSE_list = []
index_test_MSE_list = []
index_norm_MSE_list = []
index_rel_err_list = []
index_rel_err_inf_list = []

middle_train_MSE_list = []
middle_test_MSE_list = []
middle_norm_MSE_list = []
middle_rel_err_list = []
middle_rel_err_inf_list = []

middle_phal_train_MSE_list = []
middle_phal_test_MSE_list = []
middle_phal_norm_MSE_list = []
middle_phal_rel_err_list = []
middle_phal_rel_err_inf_list = []

ring_train_MSE_list = []
ring_test_MSE_list = []
ring_norm_MSE_list = []
ring_rel_err_list = []
ring_rel_err_inf_list = []

for folder in folders:
    file_path = path + folder + file
    with open(file_path) as f:
        data = f.readlines()
        for line in data:
            element = json.loads(line)
            if 'index' in element['train_set'][0]:
                index_train_MSE_list.append(element['train_MSE'])
                index_test_MSE_list.append(element['validation_MSE'])
                index_norm_MSE_list.append(element['validation_norm_MSE'])
                index_rel_err_list.append(element['relative_error'])
                index_rel_err_inf_list.append(element['relative_error_small_forces'])
            elif 'middle' in element['train_set'][0]:
                if '3rd-phal' in element['train_set'][0]:
                    middle_phal_train_MSE_list.append(element['train_MSE'])
                    middle_phal_test_MSE_list.append(element['validation_MSE'])
                    middle_phal_norm_MSE_list.append(element['validation_norm_MSE'])
                    middle_phal_rel_err_list.append(element['relative_error'])
                    middle_phal_rel_err_inf_list.append(element['relative_error_small_forces'])
                else:
                    middle_train_MSE_list.append(element['train_MSE'])
                    middle_test_MSE_list.append(element['validation_MSE'])
                    middle_norm_MSE_list.append(element['validation_norm_MSE'])
                    middle_rel_err_list.append(element['relative_error'])
                    middle_rel_err_inf_list.append(element['relative_error_small_forces'])
            elif 'ring' in element['train_set'][0]:
                ring_train_MSE_list.append(element['train_MSE'])
                ring_test_MSE_list.append(element['validation_MSE'])
                ring_norm_MSE_list.append(element['validation_norm_MSE'])
                ring_rel_err_list.append(element['relative_error'])
                ring_rel_err_inf_list.append(element['relative_error_small_forces'])

    print('====================', folder, '====================')

    tip_val_mean = np.mean(index_test_MSE_list + middle_test_MSE_list + ring_test_MSE_list)
    tip_train_mean = np.mean(index_train_MSE_list + middle_train_MSE_list + ring_train_MSE_list)
    tip_norm_mean = np.mean(index_norm_MSE_list + middle_norm_MSE_list + ring_norm_MSE_list)
    tip_rel_err_mean = np.mean(index_rel_err_list + middle_rel_err_list + ring_rel_err_list)
    tip_rel_err_std = np.std(index_rel_err_list + middle_rel_err_list + ring_rel_err_list)
    tip_rel_err_inf_mean = np.mean(index_rel_err_inf_list + middle_rel_err_inf_list + ring_rel_err_inf_list)
    tip_rel_err_inf_std = np.std(index_rel_err_inf_list + middle_rel_err_inf_list + ring_rel_err_inf_list)

    print('----Tip: ----')
    print('train MSE: ', tip_train_mean)
    print('test MSE: ', tip_val_mean)
    print('norm MSE: ', tip_norm_mean)
    print('rel err: ', tip_rel_err_mean)
    print('re_err std: ', tip_rel_err_std)
    print('rel err inf: ', tip_rel_err_inf_mean)
    print('rel err inf std: ', tip_rel_err_inf_std)

    print('----Pahls: ----')
    print('train MSE: ', np.mean(middle_phal_train_MSE_list))
    print('test MSE: ', np.mean(middle_phal_test_MSE_list))
    print('norm MSE: ', np.mean(middle_phal_norm_MSE_list))
    print('rel err: ', np.mean(middle_phal_rel_err_list))
    print('re_err std: ', np.std(middle_phal_rel_err_list))
    print('rel err inf: ', np.mean(middle_phal_rel_err_inf_list))
    print('rel err inf std: ', np.std(middle_phal_rel_err_inf_list))

    # print('----Index: ----')

    # print ('train MSE: ')
    # print('avg: ')
    # print('std: ', np.std(index_train_MSE_list))
    # print('test MSE: ')
    # print('avg: ', np.mean(index_test_MSE_list))
    # print('std: ', np.std(index_test_MSE_list))
    # print('norm MSE: ')
    # print('avg: ', np.mean(index_norm_MSE_list))
    # print('std: ', np.std(index_norm_MSE_list))
    # print('rel err: ')
    # print('avg: ', np.mean(index_rel_err_list))
    # print('std: ', np.std(index_rel_err_list))
    # print('rel err inf: ')
    # print('avg: ', np.mean(index_rel_err_inf_list))
    # print('std: ', np.std(index_rel_err_inf_list))

    # print('----Middle: ----')
    # print ('train MSE: ')
    # print('avg: ', np.mean(middle_train_MSE_list))
    # print('std: ', np.std(middle_train_MSE_list))
    # print('test MSE: ')
    # print('avg: ', np.mean(middle_test_MSE_list))
    # print('std: ', np.std(middle_test_MSE_list))
    # print('norm MSE: ')
    # print('avg: ', np.mean(middle_norm_MSE_list))
    # print('std: ', np.std(middle_norm_MSE_list))
    # print('rel err: ')
    # print('avg: ', np.mean(middle_rel_err_list))
    # print('std: ', np.std(middle_rel_err_list))
    # print('rel err inf: ')
    # print('avg: ', np.mean(middle_rel_err_inf_list))
    # print('std: ', np.std(middle_rel_err_inf_list))

    # print('----Middle Phal: ----')
    # print ('train MSE: ')
    # print('avg: ', np.mean(middle_phal_train_MSE_list))
    # print('std: ', np.std(middle_phal_train_MSE_list))
    # print('test MSE: ')
    # print('avg: ', np.mean(middle_phal_test_MSE_list))
    # print('std: ', np.std(middle_phal_test_MSE_list))
    # print('norm MSE: ')
    # print('avg: ', np.mean(middle_phal_norm_MSE_list))
    # print('std: ', np.std(middle_phal_norm_MSE_list))
    # print('rel err: ')
    # print('avg: ', np.mean(middle_phal_rel_err_list))
    # print('std: ', np.std(middle_phal_rel_err_list))
    # print('rel err inf: ')
    # print('avg: ', np.mean(middle_phal_rel_err_inf_list))
    # print('std: ', np.std(middle_phal_rel_err_inf_list))

    # print('----Ring: ----')
    # print ('train MSE: ')
    # print('avg: ', np.mean(ring_train_MSE_list))
    # print('std: ', np.std(ring_train_MSE_list))
    # print('test MSE: ')
    # print('avg: ', np.mean(ring_test_MSE_list))
    # print('std: ', np.std(ring_test_MSE_list))
    # print('norm MSE: ')
    # print('avg: ', np.mean(ring_norm_MSE_list))
    # print('std: ', np.std(ring_norm_MSE_list))
    # print('rel err: ')
    # print('avg: ', np.mean(ring_rel_err_list))
    # print('std: ', np.std(ring_rel_err_list))
    # print('rel err inf: ')
    # print('avg: ', np.mean(ring_rel_err_inf_list))
    # print('std: ', np.std(ring_rel_err_inf_list))

    index_train_MSE_list = []
    index_test_MSE_list = []
    index_norm_MSE_list = []
    index_rel_err_list = []
    index_rel_err_inf_list = []

    middle_train_MSE_list = []
    middle_test_MSE_list = []
    middle_norm_MSE_list = []
    middle_rel_err_list = []
    middle_rel_err_inf_list = []

    middle_phal_train_MSE_list = []
    middle_phal_test_MSE_list = []
    middle_phal_norm_MSE_list = []
    middle_phal_rel_err_list = []
    middle_phal_rel_err_inf_list = []

    ring_train_MSE_list = []
    ring_test_MSE_list = []
    ring_norm_MSE_list = []
    ring_rel_err_list = []
    ring_rel_err_inf_list = []