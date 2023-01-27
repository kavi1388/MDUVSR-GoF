import os
import torch
from PIL import Image, ImageOps
print(torch.__version__)
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from patchify import patchify
from downsample import *
from tkinter import Tcl

# !pip install patchify

# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("path", type=str, help="HR Path")
parser.add_argument("scale", type=int, help="res_hr_patches path")
parser.add_argument("batch_size", type=int, help="batch size")
parser.add_argument("workers", type=int, help="workers")
parser.add_argument("data_size", type=int, help="data size")

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Load Data as Numpy Array
# def read_data(path,scale,data_size):
#     hr_data = []
#     lr_data = []
#     patch = []
#     for dirname, _, filenames in os.walk(path):
#         files = Tcl().call('lsort', '-dict', filenames)
#         for filename in files:
#             if len(hr_data)<data_size:
#                 f = os.path.join(dirname, filename)
#     #             print(f)
#                 if filename.split('.')[-1] == 'png' or filename.split('.')[-1] == 'jpg':
#                     img = Image.open(f)
#                     img.load()
#                     img_array = np.asarray(img)
#                     img_array_lr = downsample(img_array, scale)
#                     img_array_lr = np.swapaxes(img_array_lr,
#                                             np.where(np.asarray(img_array_lr.shape) == min(img_array_lr.shape))[0][0], 0)
#                     lr_data.append(img_array_lr)
#
#                     img_array = np.swapaxes(img_array, np.where(np.asarray(img_array.shape) == min(img_array.shape))[0][0], 0)
#                     hr_data.append(img_array)
#
#     return hr_data, lr_data


def read_data(hr_path,lr_path):
    hr_data = []
    lr_data = []

    for dirname, _, filenames in os.walk(hr_path):
        files = Tcl().call('lsort', '-dict', filenames)
        for filename in files:
            f = os.path.join(dirname, filename)
#             print(f)
            if filename.split('.')[-1] == 'png' or filename.split('.')[-1] == 'jpg':
                img = Image.open(f)
                img.load()
                img_array = np.asarray(img)
                img_array = np.swapaxes(img_array,
                                        np.where(np.asarray(img_array.shape) == min(img_array.shape))[0][0], 0)
                hr_data.append(img_array)

                for dirname, _, filenames in os.walk(lr_path):
                    files = Tcl().call('lsort', '-dict', filenames)
                    for filename in files:
                        f = os.path.join(dirname, filename)
                        #             print(f)
                        if filename.split('.')[-1] == 'png' or filename.split('.')[-1] == 'jpg':
                            img = Image.open(f)
                            img.load()
                            img_array = np.asarray(img)
                            img_array = np.swapaxes(img_array,np.where(np.asarray(img_array.shape) == min(img_array.shape))[0][0],
                                                    0)
                            lr_data.append(img_array)
    return hr_data, lr_data

# parameters to the function are lr_frames, corresponding hr frames, group_of_frames as input, max_len for hr frames
#(argument given in data_load function)

# parameters to the function are lr_frames, corresponding hr frames, group_of_frames as input, max_len for hr frames
#(argument given in data_load function)

class CustomDataset(Dataset):
    def __init__(self, image_data, labels, gof, max_len):
        self.gof = gof
        self.max_len = max_len//self.gof
        self.image_data = image_data
        self.labels = labels

    def __len__(self):
        return (self.max_len)

    def __getitem__(self, index):

        image = self.image_data[(index*self.gof):(index*self.gof) + self.gof]/255.0
        label = self.labels[(index*self.gof) + self.gof//2]/255.0
        return (
        torch.tensor(image, dtype=torch.float),
        torch.tensor(label, dtype=torch.float)
        )


def data_load(all_lr_data, all_hr_data, batch_size, workers, gof):
    # Train, Test, Validation splits
    train_data_hr = all_hr_data[:3 * len(all_hr_data) // 4]
    train_data_lr = all_lr_data[:3 * len(all_lr_data) // 4]
    print(f'len of train hr data ={len(train_data_hr)}')

    val_data_hr = all_hr_data[3 * len(all_hr_data) // 4:len(all_hr_data)]
    val_data_lr = all_lr_data[3 * len(all_lr_data) // 4:len(all_lr_data)]

    print(f'len of val hr data ={len(val_data_hr)}')
    # test_data_hr = all_hr_data[(len(all_hr_data)//3)+(len(all_hr_data)//4):(len(all_hr_data)//3)+(len(all_hr_data)//2)]
    # test_data_lr = all_lr_data[(len(all_lr_data)//3)+(len(all_lr_data)//4):(len(all_lr_data)//3)+(len(all_lr_data)//2)]

    print(f'hr {len(all_hr_data)}')
    print(f'lr {len(all_lr_data)}')

    train_data = CustomDataset(np.asarray(train_data_lr), np.asarray(train_data_hr), gof,
                               len(np.asarray(train_data_lr)))
    val_data = CustomDataset(np.asarray(val_data_lr), np.asarray(val_data_hr), gof, len(np.asarray(val_data_lr)))
    # test_data = CustomDataset(np.asarray(test_data_lr),np.asarray(test_data_hr))
    print(f'dataset created')
    # Load Data as Numpy Array
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers)

    print(f'train {len(train_data_hr)}')
    print(f'val {len(val_data_hr)}')
    # print(f'test {len(test_data_hr)}')

    return train_loader, val_loader