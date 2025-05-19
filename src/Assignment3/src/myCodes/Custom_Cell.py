import torch
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size,kernel_size=3,padding=1, bias=True):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: int
            Number of channels of input tensor.
        hidden_size: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Linear layer to compute all gates at once
        # Note: 4 * hidden_size for i, f, o, g (input, forget, output, candidate)
        
        self.f_t = nn.Conv1d(in_channels=input_size + hidden_size,
        out_channels=hidden_size,  # Matches hidden state dim
        kernel_size=kernel_size,
        padding=padding,
        bias=bias # To maintain spatial dims
        )
        self.i_t =nn.Conv1d(in_channels=input_size + hidden_size,
        out_channels=hidden_size,  # Matches hidden state dim
        kernel_size=kernel_size,
        padding=padding,
        bias=bias # To maintain spatial dims
        )

        self.c_hat_t =nn.Conv1d(in_channels=input_size + hidden_size,
        out_channels=hidden_size,  # Matches hidden state dim
        kernel_size=kernel_size,
        padding=padding,
        bias=bias # To maintain spatial dims
        )
        
        self.o_t = nn.Conv1d(in_channels=input_size + hidden_size,
        out_channels=hidden_size,  # Matches hidden state dim
        kernel_size=kernel_size,
        padding=padding,
        bias=bias # To maintain spatial dims
        )

        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        
        gates = self.linear(combined)  # Shape: (batch, 4 * hidden_size)
        
        # Split into input, forget, output, and candidate gates
        #cc_i, cc_f, cc_o, cc_g = gates.chunk(4, dim=1)  # Each shape: (batch, hidden_size)
        
        f_t=torch.sigmoid(self.f_t(combined).T)
        i_t=torch.sigmoid(self.i_t(combined).T)
        c_hat_t=torch.tanh(self.c_hat_t(combined).T)
        o_t=torch.sigmoid(self.o_t(combined).T)
        
        c_t=f_t*c_cur+i_t*c_hat_t
        h_t=o_t*torch.tanh(c_t)
        
        return h_t, c_t

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_size, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_size, height, width, device=self.conv.weight.device))
