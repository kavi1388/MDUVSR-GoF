import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn

# Original ConvLSTM cell as proposed by Shi et al.
class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels,
    kernel_size, padding, activation, frame_size):

        super(ConvLSTMCell, self).__init__()

#         print(f'in_channels, out_channels, kernel_size, padding, activation, frame_size {in_channels, out_channels,kernel_size, padding, activation, frame_size}')

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding)

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):

#         print(f'H_prev = {H_prev.shape}')
#         print(f'X = {X.shape}')

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

#         print(f'conv_output {conv_output.shape}')

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

#         print(f'i_conv = {i_conv.shape}')
#         print(f'f_conv = {f_conv.shape}')

#         print(f'self.W_ci = {self.W_ci.shape}')
#         print(f'self.W_cf = {self.W_cf.shape}')
#         print(f'C_prev = {C_prev.shape}')

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C

class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels,
    kernel_size, padding, activation, frame_size):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels,
        kernel_size, padding, activation, frame_size)

    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, seq_len,_, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, seq_len,  self.out_channels,
        height, width, device=device)

        # Initialize Hidden State
        H = torch.rand(batch_size, self.out_channels,
        height, width, device=device)

        # Initialize Cell Input
        C = torch.rand(batch_size,self.out_channels,
        height, width, device=device)

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.convLSTMcell(X[:,time_step,:], H, C)

            output[:,time_step,:] = H

        return output