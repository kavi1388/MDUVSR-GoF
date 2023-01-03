import warnings
warnings.filterwarnings('ignore')
from PIL import Image, ImageOps
import numpy as np # linear algebra
import os # accessing directory structure
from scipy.ndimage import gaussian_filter
import argparse
import cv2

def downsample(img, scale):
    img = cv2.GaussianBlur(img, (0,0), 1, 1)   # func args = img, kernel_size, X_sigma, Y_sigma

    #     downsample_shape=(img.shape[0]//scale,img.shape[1]//scale,img.shape[2])

    im2 = Image.fromarray(np.uint8(img)).resize((img.shape[1] // scale, img.shape[0] // scale), Image.BICUBIC)
    lr_bicubic = np.asarray(im2)

    return lr_bicubic

def gen_lr(frame_path,lr_path,scale):

    for dirname, _, filenames in os.walk(frame_path):
        for filename in filenames:
            f = os.path.join(dirname, filename)
            print(filename)
            if filename.split('.')[-1] == 'jpg':
                img = Image.open(f)
                img.load()
                img_array = np.asarray(img)
                img_array_lr = downsample(img_array,scale)
                # print(type(Image.fromarray(np.uint8(img_array_lr))))
                Image.fromarray(np.uint8(img_array_lr)).save(f"{lr_path}/{filename.split('.')[0]}_lr.jpg")
