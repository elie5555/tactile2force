import preprocessing.pp_sequence as pp
import dataset.save_dataset as sd
from dataset.save_dataset import save_dataset
'''
files = ['middle_tip_2024-05-17-17-13-23', 'ring_tip_2024-05-22-08-33-52', 'middle_tip_2024-05-22-08-31-18', 'index_tip_17_05_24_1',
 'middle_tip_2024-05-21-12-14-09', 'index_tip_2024-05-17-16-54-09', 'ring_tip_2024-05-21-16-46-14', 'ring_tip_2024-05-21-16-44-25',
  'ring_tip_2024-05-21-16-42-10', 'middle_tip_2024-05-21-12-18-27' ,'index_tip_17_05_24_2', 'index_tip_2024-05-22-08-28-24',
 'index_tip_07_05_24_2', 'index_tip_07_05_24_1', 'middle_tip_07_05_24_1', 'middle_tip_07_05_24_2', 
 'ring_tip_07_05_24_1', 'ring_tip_07_05_24_2', 'middle-3rd-phal_2024-06-04-09-05-18']
'''
#files = ['middle-3rd-phal_2024-06-18-15-47-28', 'middle-3rd-phal_2024-06-18-15-52-18',
#          'middle-3rd-phal_2024-06-18-15-54-52']

files = ["index_tip_2024-06-21-12-39-30"]

for file_name in files:
    string = file_name.split('_')[0]
    if string == 'index':
        patch_of_interest = '2'
    elif string == 'middle':
        patch_of_interest = '5'
    elif string == 'ring':
        patch_of_interest = '8'
    elif string == 'thumb':
        patch_of_interest = '0'
    elif string == 'middle-3rd-phal':
        patch_of_interest = '6A'

    input_bag = "data/"+file_name+".bag"
    output_pickle = "pickle_data/"+file_name

    if file_name == 'middle_tip_2024-05-17-17-13-23':
        pp.do_it(input_bag, output_pickle, patch_of_interest, verbose=True, synchronize=False, cut_data=True, cut_time=23.5)
    else:
        pp.do_it(input_bag, output_pickle, patch_of_interest, verbose=True, synchronize=False)

    save_dataset(file_name)