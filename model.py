"""### Final Model"""
import torch
import torch.nn as nn
from DefConv2D import *
from ConvLSTM import *

# +
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class mdpvsr_1defconv(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding,
                 scale, activation, frame_size, group_of_frames, state=None):
        super(mdpvsr_1defconv, self).__init__()

        print(
            f'num_channels, num_kernels, group_of_frames, kernel_size, padding, scale, activation, frame_size {num_channels, num_kernels, group_of_frames, kernel_size, padding, scale, activation, frame_size}')
        self.group_of_frames = group_of_frames
        self.convlstm1 = ConvLSTM(
            in_channels=num_channels, out_channels=num_kernels,
            kernel_size=kernel_size, padding=padding,
            activation=activation, frame_size=frame_size)

        self.deformable_convolution1 = DeformableConv2d(
            in_channels=group_of_frames * num_kernels,
            out_channels=num_kernels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=padding[0],
            bias=True)

        # Add rest of the layers

        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels * scale ** 2,
            kernel_size=kernel_size, padding=padding)

        self.up_block = nn.PixelShuffle(scale)

    def forward(self, X, state=None):
        # Forward propagation through all the layers

        lr = torch.transpose(X, 1, 2)
        #         plt.figure(figsize=(15, 10))
        #         plt.title('Input')
        #         plt.imshow(lr.cpu()[-1].detach().numpy()[:,0,:,:].T.astype(int))
        #         print(f'Input {lr.shape}')
        output_convlstm = self.convlstm1(lr)
        #         print(f'output_convlstm shape {output_convlstm.shape}')
        x = output_convlstm
        #         plt.figure(figsize=(15, 10))
        #         plt.title('after convlstm')
        #         plt.imshow(x.cpu()[-1].detach().numpy()[:,0,:,:].T.astype(int))
        y = list(torch.chunk(x, self.group_of_frames, dim=1))
        for i in range(len(y)):
            y[i] = torch.squeeze(y[i])
        x = torch.cat(y, dim=1)
        #         plt.figure(figsize=(15, 10))
        #         plt.title('before def conv')
        #         plt.imshow(x.cpu()[-1][-1].detach().numpy().T.astype(int))
        #         print(f'input to def conv shape {x}')
        x = self.deformable_convolution1(x)
        #         plt.figure(figsize=(15, 10))
        #         plt.title('after def conv')
        #         plt.imshow(x.cpu()[-1][-1].detach().numpy().T.astype(int))
        #         print(f'output from def conv {x}')
        #         print(f'torch.chunk(x,self.group_of_frames,dim=1) {torch.chunk(x,self.group_of_frames,dim=1)[0].shape}')
        #         x= torch.stack(torch.chunk(x,self.group_of_frames,dim=1),dim=1)
        #         print(f'torch.stack(x,dim=1) {x.shape}')
        #         x = torch.cat((x,lr),2)
        #         print(f'input to conv shape {x.shape}')
        x = self.conv(x)
        #         plt.figure(figsize=(15, 10))
        #         plt.title('before up conv')
        #         plt.imshow(x.cpu()[-1][-1].detach().numpy().T.astype(int))
        #         print(f'input to up_block {x}')
        output = self.up_block(x)
        #         plt.figure(figsize=(15, 10))
        #         plt.title('After up conv')
        #         plt.imshow(output.cpu()[-1].detach().numpy().T.astype(int))
        #         print(f'output {output}')

        output = torch.clamp(output, 0, 255)
        #         plt.figure(figsize=(15, 10))
        #         plt.title('final output')
        #         plt.imshow(output.cpu()[-1].detach().numpy().T.astype(int))
        #         print('output')
        #         print(output)

        return output