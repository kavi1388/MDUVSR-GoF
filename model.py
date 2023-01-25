"""### Final Model"""
import torch
import torch.nn as nn
from DefConv2D import *
from ConvLSTM import *
import gc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class mdpvsr_1defconv(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding,
                 scale, activation, frame_size, group_of_frames, num_layers, state=None):
        super(mdpvsr_1defconv, self).__init__()

        print(
            f'num_channels, num_kernels, group_of_frames, kernel_size, padding, scale, activation, frame_size {num_channels, num_kernels, group_of_frames, kernel_size, padding, scale, activation, frame_size}')
        self.group_of_frames = group_of_frames
        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        )

        # Add rest of the layers
        for l in range(2, num_layers + 1):
            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding,
                    activation=activation, frame_size=frame_size)
            )

            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
            )

        self.deformable_convolution1 = DeformableConv2d(
            in_channels=group_of_frames * num_channels,
            out_channels=group_of_frames * num_channels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=padding[0],
            bias=True)

        # Add rest of the layers

        self.conv1 = nn.Conv3d(
            in_channels=num_kernels, out_channels=num_channels, kernel_size=(1, 1, 1), padding=0)

        self.conv2 = nn.Conv3d(
            in_channels=num_channels, out_channels=num_channels * scale ** 2, kernel_size=(1, 1, 1), padding=0)

        self.up_block = nn.PixelShuffle(scale)

    def forward(self, X, state=None):
        lr = torch.transpose(X, 1, 2)
        x = self.sequential(lr)
        x = self.conv1(x)
        y = list(torch.chunk(x, self.group_of_frames, dim=1))
        for i in range(len(y)):
            y[i] = torch.squeeze(y[i])
        x = torch.cat(y, dim=1)
        x = self.deformable_convolution1(x)
        x = torch.chunk(x, self.group_of_frames, dim=1)
        x = torch.stack(x, dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.conv2(x)
        x = torch.transpose(x, 1, 2)

        output = self.up_block(x)
        return nn.Sigmoid()(output)