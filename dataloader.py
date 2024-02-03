import pickle as pkl
import torch
import numpy as np
import math
# import cv2 
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms, datasets

from GLOBAL_VARS import * 

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

class ImageDataset(Dataset):
    def __init__(self, type_val_train ):
        """
        Args:
        """
        # transform definitino
        self.transforms =  transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

        # open split file
        save_file = "./train_val_split.pkl"


        with open(save_file, 'rb') as f:
            data = pkl.load(f)
        
        if type_val_train == "train":
            self.files = data['train_files']
            self.labels = data['train_labels']

        else:
            self.files = data['val_files']
            self.labels = data['val_labels']

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
            Get the data corresponding to the idx
        """
        file = self.files[idx]
        label = self.labels[idx]

        image = np.asarray(Image.open(file))
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = np.asarray(image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH)))

        image = self.transforms(resize_image)
        ret_dict = {
            'image': image,
            'label': torch.tensor(label, dtype = torch.long)
        }
        return ret_dict


if __name__ == "__main__":
    dataset_new = ImageDataset('val')
    dataset_new[0]