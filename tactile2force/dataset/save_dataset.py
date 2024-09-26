import json

def save_dataset(file_name):

    finger = file_name.split('_')[0]
    if 'index' in finger:
        finger = 'index'
    elif 'middle' in finger:
        finger = 'middle'
    elif 'ring' in finger:
        finger = 'ring'
    elif 'thumb' in finger:
        finger = 'thumb'

    file_name = f"./pickle_data/{file_name}/"

    section = None
    if 'tip' in file_name:
        section = 'tips'
    elif 'phal' in file_name:
        section = 'phal'

    with open("./dataset/datasets.json", 'r') as f:
        data = json.load(f)
        data[section][finger].append(file_name)

    with open("./dataset/datasets.json", 'w') as f:
        json.dump(data, f)
        
        print(file_name, " saved to datasets.json")