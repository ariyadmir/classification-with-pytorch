import pickle as pkl
import random 
import torch
import numpy as np
import math
# import cv2 
from PIL import Image

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from GLOBAL_VARS import * 

def filter_csv(train_files, class_labels):
    ret_files = []
    ret_labels = []

    for i, file in enumerate(train_files):
        if not '.csv' in file:
            ret_files.append(file)
            ret_labels.append(class_labels[i])

    return ret_files, ret_labels

def index_list(input_list, indices):
    new_list = [input_list[i] for i in indices]

    return new_list

if __name__ == "__main__":
    folders = os.listdir(train_path)

    train_number = []
    class_num = []

    all_files = []
    all_class_labels =[]

    for folder in folders:
        train_files = os.listdir(train_path + '/' + folder)
        for file in train_files:
            all_files.append(train_path + '/' + folder + '/' + file)
            all_class_labels.append(int(folder))

    files_no_csv, filter_labels_no_csv = filter_csv(all_files, all_class_labels)
    
    # import ipdb
    # ipdb.set_trace()

    seventy_percent = 0.7 * len(files_no_csv)

    mylist = []
    nums = 0
    while(True):
        x = random.randint(0, len(files_no_csv)-1)
        if not(x in mylist):
            mylist.append(x)
            nums += 1
        
        if nums == seventy_percent:
            break

    invert_list = list(set(list(range(0, len(files_no_csv)))).difference(set(mylist)))
    # invert_list = list(range(0, len(files_no_csv)))
    print(np.max(mylist))
    print(len(filter_labels_no_csv))

    save_dict = {
        'train_files': index_list(files_no_csv, mylist),
        'train_labels': index_list(filter_labels_no_csv, mylist),

        'val_files': index_list(files_no_csv, invert_list),
        'val_labels': index_list(filter_labels_no_csv, invert_list)
    }

    save_file = "./train_val_split.pkl"

    with open(save_file, 'wb') as f:
        pkl.dump(save_dict, f)

    import ipdb
    ipdb.set_trace()

# Want:
# 1. train and val splits with their class labels