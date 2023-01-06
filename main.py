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

"""### Preparing Data"""
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import cv2


# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument


# parser.add_argument("model", type=str, help="model to use")
parser.add_argument("path", type=str, help="HR Path")
parser.add_argument("data_size", type=int, help="data size")
# parser.add_argument("lr_data", type=str, help="LR Path")
parser.add_argument("batch_size", type=int, help="batch size")
parser.add_argument("gof", type=int, help="group of frames")
parser.add_argument("workers", type=int, help="workers")
parser.add_argument("result", type=str, help="result Path (to save)")
parser.add_argument("scale", type=int, help="downsampling scale")
parser.add_argument("epochs", type=int, help="epochs")
parser.add_argument("name", type=str, help="model name")
# Read arguments from command line
args = parser.parse_args()

# model_to_use = args.model
res_path = args.result
scale = args.scale
epochs = args.epochs
name = args.name
hr_path = args.path
data_size = args.data_size
# lr_path = args.lr_data
batch_size = args.batch_size
workers = args.workers
gof = args.gof

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(res_path):
    os.makedirs(res_path)

all_hr_data, all_lr_data = read_data(hr_path, scale, data_size)
# all_lr_data = read_data(lr_path)
print('read')
train_loader, val_loader = data_load(all_lr_data,all_hr_data, batch_size, workers, gof)
print('loaded')

# ### Defining Model

print('Computation device: ', device)

# +
# if model_to_use == 'mdpvsr':
#     model = mdpvsr(num_channels=train_loader.dataset[0][0].shape[0], num_kernels=train_loader.dataset[0][0].shape[1]//2,
#                kernel_size=(3,3), padding=(1, 1), scale=scale).to(device)

# +
# elif model_to_use == 'mduvsr_6defconv':
#     model = mduvsr_6defconv(num_channels=train_loader.dataset[0][0].shape[0], num_kernels=train_loader.dataset[0][0].shape[1]//2,
#                kernel_size=(3,3), padding=(1, 1), scale=scale).to(device)

# +
# elif model_to_use == 'mduvsr_6defconv_pixelshuff':
#     model = mduvsr_6defconv_pixelshuff(num_channels=train_loader.dataset[0][0].shape[0], num_kernels=train_loader.dataset[0][0].shape[1] // 2,
#     kernel_size=(3, 3), padding = (1, 1), scale = scale).to(device)

# +
# elif model_to_use == 'mduvsr_1defconv':
#     model = mduvsr_1defconv(num_channels=train_loader.dataset[0][0].shape[0], num_kernels=train_loader.dataset[0][0].shape[1] // 2,
#     kernel_size=(3, 3), padding = (1, 1), scale = scale).to(device)

# +
# elif model_to_use == 'mduvsr_2defconv':
#     model = mduvsr_2defconv(num_channels=train_loader.dataset[0][0].shape[0], num_kernels=train_loader.dataset[0][0].shape[1] // 2,
#     kernel_size=(3, 3), padding = (1, 1), scale = scale).to(device)
# -

# elif model_to_use == 'mdpvsr_1defconv':
# elif model_to_use == 'mdpvsr_1defconv':
model = mdpvsr_1defconv(num_channels=train_loader.dataset[0][0].shape[1], num_kernels=train_loader.dataset[0][0].shape[1],
               kernel_size=(3, 3), padding=(1,1), activation="relu", scale=4, group_of_frames = gof,
            frame_size=(train_loader.dataset[0][0].shape[2],train_loader.dataset[0][0].shape[3])).to(device)# +
# elif model_to_use == 'mdpvsr_2defconv':
#     model = mdpvsr_2defconv(num_channels=train_loader.dataset[0][0].shape[0], num_kernels=train_loader.dataset[0][0].shape[1] // 2,
#     kernel_size=(3, 3), padding = (1, 1), scale = scale).to(device)

# +
# else:
#     model = mduvsr(num_channels=train_loader.dataset[0][0].shape[0],
#                    num_kernels=train_loader.dataset[0][0].shape[1] // 2,
#                    kernel_size=(3, 3), padding=(1, 1), scale=scale).to(device)
# -

print(model)
print(summary(model, (train_loader.dataset[0][0].shape)))

"""### Loss Function"""


def get_outnorm(x: torch.Tensor, out_norm: str = '') -> torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
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


"""### Training"""

scaler = torch.cuda.amp.GradScaler()
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

# criterion = nn.L1Loss()
criterion = CharbonnierLoss()
num_epochs = epochs

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
        if batch_num % 200 ==0:
            print(f'batch_num {batch_num}')
        output = model(input.cuda())
        loss = criterion(output, target.cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        try:
            psnr.append(piq.psnr(output.cpu(), target, data_range=255., reduction='mean'))
        except:
            psnr.append(0)
        try:
            ssim.append(piq.ssim(output.cpu(), target, data_range=255.))
        except:
            ssim.append(0)
        # lpips.append(piq.LPIPS(reduction='mean')(torch.clamp(output, 0, 1), torch.clamp(target.cuda(), 0, 255)))
        torch.cuda.empty_cache()
    train_loss /= len(train_loader.dataset)
    psnr_avg= sum(psnr)/len(train_loader.dataset)
    ssim_avg= sum(ssim)/len(train_loader.dataset)
    # lpips_avg= sum(lpips)/len(train_loader.dataset)
    psnr_max = max(psnr)
    ssim_max = max(ssim)
    
    if ssim_avg > ssim_best:
        ssim_best = ssim_avg
        params = f'{epoch} , scale={scale} ,ssim = {ssim_best} ,{name}'
        PATH = f'{res_path}/{params}.pth'
        torch.save(model.state_dict(), PATH)
        
    # lpips_max = max(lpips)
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for input, target in val_loader:
            count+=1
            output = model(input.cuda())
            loss = criterion(output, target.cuda())
            try:
                psnr_test.append(piq.psnr(output.cpu(), target, data_range=255., reduction='mean'))
            except:
                psnr_test.append(0)
            try:
                ssim_test.append(piq.ssim(output.cpu(), target, data_range=255.))
            except:
                ssim_test.append(0)
            # lpips_test.append(piq.LPIPS(reduction='mean')(torch.clamp(output, 0, 1), torch.clamp(target.cuda(), 0, 255)))
            val_loss += loss.item()
            
            if count % 500 == 0 and ssim_avg == ssim_best:
                if not os.path.exists(f'{res_path}/{params}'):
                    os.makedirs(f'{res_path}/{params}')
                plt.figure(figsize=(40, 10))
                plt.subplot(141)
                plt.title('Input')
                plt.imshow(input.cpu()[-1].detach().numpy().T.astype(int))
                plt.subplot(142)
                plt.title('Our Result')
                plt.imshow(output.cpu()[-1].detach().numpy().T.astype(int))
                plt.subplot(143)
                plt.title('Target')
                plt.imshow(target.cpu()[-1].detach().numpy().T.astype(int))
                plt.subplot(144)
                plt.title('Bicubic')
                plt.imshow(cv2.resize(input.cpu()[-1].detach().numpy().T, (target.shape[2],target.shape[3]), interpolation= cv2.INTER_LINEAR).astype(int))
                plt.savefig(f"{res_path}/{params}/psnr_{piq.psnr(output.cpu(), target, data_range=255., reduction='mean')} "
                            f"and ssim_{piq.ssim(output.cpu(), target, data_range=255.)}.png", bbox_inches="tight",
                            pad_inches=0.0)


    val_loss /= len(val_loader.dataset)
    psnr_test_avg = sum(psnr_test)/len(val_loader.dataset)
    ssim_test_avg = sum(ssim_test)/len(val_loader.dataset)
    # lpips_test_avg = sum(lpips_test)/len(val_loader.dataset)
    psnr_test_max = max(psnr_test)
    ssim_test_max = max(ssim_test)
    # lpips_test_max = max(lpips_test)

    print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f} in {:.2f} and SSIM\n".format(
        epoch+1, train_loss, val_loss, time.time()-st))
    print(f'Train PSNR avg {torch.round(psnr_avg)}, PSNR max {torch.round(psnr_max)} and Test PSNR avg {torch.round(psnr_test_avg)}, test PSNR max {torch.round(psnr_test_max)}')
    print(f'Train SSIM avg {torch.round(ssim_avg)} , SSIM max {torch.round(ssim_max)} and Test SSIM avg {torch.round(ssim_test_avg)}, test SSIM max {torch.round(ssim_test_max)}')




# test_loader = torch.load('test_loader.pt', map_location=torch.device('cuda'))
# model.eval()
# ssim_val = []
# psnr_val = []
# lpips_val = []
# running_psnr = 0
#
# with torch.no_grad():
#     for input, target in test_loader:
#         output = model(input.cuda())
#         psnr_val.append(piq.psnr(output, target.cuda(), data_range=255., reduction='mean'))
#         ssim_val.append(piq.ssim(output, target.cuda(), data_range=255.))
#         lpips_val.append(piq.LPIPS(reduction='mean')(torch.clamp(output, 0, 1), torch.clamp(target.cuda(), 0, 255)))
#
#         print(f'psnr value ={psnr_val[-1]}')
#         print(f'ssim value ={ssim_val[-1]}')
#         print(f'lpips value ={lpips_val[-1]}')
#
#     with open(r'name_quality metrics', 'w') as fp:
#         fp.write("\n PSNR")
#         for item in psnr_val:
#             # write each item on a new line
#             fp.write("%s\n" % item)
#
#         fp.write("\n SSIM")
#         for item in ssim_val:
#             # write each item on a new line
#             fp.write("%s\n" % item)
#
#         fp.write("\n LPIPS")
#         for item in lpips_val:
#             # write each item on a new line
#             fp.write("%s\n" % item)


