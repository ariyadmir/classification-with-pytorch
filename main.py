import numpy as np
# import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from PIL import Image

from GLOBAL_VARS import *

from networks import Classifier

from torch.utils.data import DataLoader

from dataloader import ImageDataset
from dataloader_test import TestImageDataset

train_set = ImageDataset('train')
val_set = ImageDataset('val')
test_set = TestImageDataset()

lr = 0.0001
epochs = 30

network = Classifier().cuda()

opt = torch.optim.Adam(lr=lr, params = network.parameters())
loss = torch.nn.CrossEntropyLoss()

dataloader_train = DataLoader(train_set, batch_size = 32, num_workers = 32)
dataloader_test = DataLoader(test_set, batch_size = 32, num_workers = 32)
dataloader_val =  iter(DataLoader(val_set, batch_size = 32, num_workers = 32))

for epoch in range(epochs):

    for i, dict in enumerate(dataloader_train):
        img = dict['image'].cuda()
        gt = dict['label'].cuda()
        opt.zero_grad()

        output = network(img)
        loss_val = loss(output, gt)

        loss_val.backward()
        opt.step()

        if i%100 == 0:
            val_dict = next(dataloader_val)
            val_img = val_dict['image'].cuda()
            val_gt = val_dict['label'].cuda()

            with torch.no_grad():
                output = network(val_img)

            validation_loss = loss(output, val_gt)
            acc = (torch.argmax(output, 1) == val_gt).float().mean()

            print("Epoch {} Iteration {}".format(epoch, i))
            print("Validation loss", validation_loss.item())
            print("Accuracy {} %".format(acc.item() * 100))

    if epoch % 2 == 0:
        all_accs = []
        for i, dict in enumerate(dataloader_test):
            img = dict['image'].cuda()
            gt = dict['label'].cuda()

            with torch.no_grad():
                output = network(img)
            acc = (torch.argmax(output, 1) == gt).float().mean()
            all_accs.append(acc.cpu().numpy())

        print('test accuracy')
        print(np.mean(all_accs))


