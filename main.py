import os
import time
import torch
from PIL import Image, ImageOps
print(torch.__version__)
import piq
from model import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchsummary import summary
from dataPrep import read_data, data_load
from piqa import ssim
from piqa.utils.functional import gaussian_kernel
"""### Preparing Data"""
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gc
import argparse
import cv2
from piqa import SSIM

# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument


parser.add_argument("path", type=str, help="HR Path")
parser.add_argument("data_size", type=int, help="data size")
parser.add_argument("batch_size", type=int, help="batch size")
parser.add_argument("gof", type=int, help="group of frames")
parser.add_argument("workers", type=int, help="workers")
parser.add_argument("result", type=str, help="result Path (to save)")
parser.add_argument("scale", type=int, help="downsampling scale")
parser.add_argument("epochs", type=int, help="epochs")
parser.add_argument("name", type=str, help="model name")
parser.add_argument("layers", type=int, help="layers")
parser.add_argument("kernels", type=int, help="kernels")
# Read arguments from command line
args = parser.parse_args()

# model_to_use = args.model
res_path = args.result
scale = args.scale
epochs = args.epochs
name = args.name
hr_path = args.path
data_size = args.data_size
batch_size = args.batch_size
workers = args.workers
gof = args.gof
layers = args.layers
kernels = args.kernels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(res_path):
    os.makedirs(res_path)

all_hr_data, all_lr_data = read_data(hr_path, scale, data_size)
print('read')
train_loader, val_loader = data_load(all_lr_data,all_hr_data, batch_size, workers, gof)
print('loaded')

# ### Defining Model

print('Computation device: ', device)

model = mdpvsr_1defconv(num_channels=train_loader.dataset[0][0].shape[1],
num_kernels=kernels,
kernel_size=(3, 3),
padding=(1,1),
activation="relu",
scale= scale,
group_of_frames = gof,
frame_size=(train_loader.dataset[0][0].shape[2],train_loader.dataset[0][0].shape[3]),
num_layers=layers).to(device)

print(model)
print(summary(model, (train_loader.dataset[0][0].shape)))

"""### Loss Function"""

from piqa import ssim
from piqa.utils.functional import gaussian_kernel


def get_outnorm(x: torch.Tensor, out_norm: str = '') -> torch.Tensor:
    #     """ Common function to get a loss normalization value. Can
    #         normalize by either the batch size ('b'), the number of
    #         channels ('c'), the image size ('i') or combinations
    #         ('bi', 'bci', etc)
    #     """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if 'b' in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if 'c' in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[-3]
    if 'i' in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1] * img_shape[-2]

    return norm


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, out_norm: str = 'bci'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm

    def forward(self, x, y):
        norm = get_outnorm(x, self.out_norm)
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps ** 2))
        return loss * norm


# class SSIMLoss(SSIM):
#     def forward(self, x, y):

#         return 1. - super().forward(x, y)

if not os.path.exists(res_path):
    os.makedirs(res_path)

scaler = torch.cuda.amp.GradScaler()
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

# criterion = nn.L1Loss()


# criterion = SSIMLoss().cuda()
criterion = CharbonnierLoss()
num_epochs = epochs
# num_epochs = 3

# Training the model

# Model training parameters
# Creating results path
if not os.path.exists(res_path):
    os.makedirs(res_path)

# Initializing Gradient Scaler
scaler = torch.cuda.amp.GradScaler()

# Initializing Optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

# Loss Criterion and Epochs
criterion = CharbonnierLoss()
num_epochs = epochs
params = 'ConvLSTM_zero_init_26th_Jan'
"""### Training"""
for epoch in range(num_epochs):
    psnr = []
    ssim = []
    lpips = []
    train_loss = 0
    ssim_best = 0
    count = 0
    psnr_test = []
    ssim_test = []
    lpips_test = []
    model.train()
    st = time.time()
    for batch_num, data in enumerate(train_loader, 0):
        input, target = data[0].to(device), data[1]
        # if batch_num % 200 == 0:
        #     print(f'batch_num {batch_num}')
        # Modifying target
        #         target = target.permute(0,1,2,4,3)
        output = model(input.cuda())
        #         print(f'output {output.size()}')
        #         print(f'target {target.size()}')
        loss = criterion(output, target.cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        psnr.append(piq.psnr(output, target.cuda(), data_range=1., reduction='mean').mean())
        ssim.append(piq.ssim(output, target.cuda(), data_range=1.))
        if count % 10 == 0:
            if not os.path.exists(f'{res_path}/{params}'):
                os.makedirs(f'{res_path}/{params}')
            plt.figure(figsize=(40, 10))
            plt.subplot(141)
            plt.title('Input')
            plt.imshow((input * 255).cpu()[-1][3].detach().numpy().T.astype(int))
            plt.subplot(142)
            plt.title('Our Result')
            plt.imshow((output * 255).cpu()[-1].detach().numpy().T.astype(int))
            plt.subplot(143)
            plt.title('Target')
            plt.imshow((target * 255).cpu()[-1].detach().numpy().T.astype(int))
            plt.subplot(144)
            plt.title('Bicubic')
            plt.imshow(cv2.resize((input * 255).cpu()[-1][3].detach().numpy().T, (target.shape[2], target.shape[3]),
                                  interpolation=cv2.INTER_CUBIC).astype(int))
            # plt.savefig(f"{res_path}/{params}/psnr_{piq.psnr(output.cpu(), target, data_range=1., reduction='mean')} "
            #             f"and ssim_{piq.ssim(output.cpu(), target, data_range=1.)}.png", bbox_inches="tight",
            #             pad_inches=0.0)
            plt.savefig(f"{res_path}/{params}/{count}_model_training_output.png", bbox_inches="tight",
                        pad_inches=0.0)
            # plt.show()
        # lpips.append(piq.LPIPS(reduction='mean')(torch.clamp(output, 0, 1), torch.clamp(target.cuda(), 0, 255)))
        torch.cuda.empty_cache()
    train_loss /= len(train_loader.dataset)
    psnr_avg = sum(psnr) / len(psnr)
    ssim_avg = sum(ssim) / len(ssim)
    # lpips_avg= sum(lpips)/len(train_loader.dataset)
    psnr_max = max(psnr)
    ssim_max = max(ssim)

    if ssim_avg > ssim_best:
        ssim_best = ssim_avg
        PATH = f'{res_path}/{params}_Epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), PATH)

    if not os.path.exists(f'{res_path}/{params}/metrics'):
        os.makedirs(f'{res_path}/{params}/metrics')
    with open(f'{res_path}/{params}/metrics/{epoch + 1}_quality metrics', 'w') as fp:

        fp.write("\n SSIM")
        for item in ssim:
            # write each item on a new line
            fp.write("%s\n" % item)
            # num += num

        fp.write("\n PSNR")
        for item in psnr:
            # write each item on a new line
            fp.write("%s\n" % item)
            # num += num

    print("Epoch:{} Training Loss:{:.6f} in {:.2f}\n".format(
        epoch+1, train_loss, time.time()-st))
    print(f'Train PSNR avg {psnr_avg}, PSNR max {psnr_max}, Train SSIM avg {ssim_avg} , SSIM max {ssim_max}')