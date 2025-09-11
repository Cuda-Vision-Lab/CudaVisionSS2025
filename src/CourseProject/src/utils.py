import shutil
import os
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class Patchifier:
    """ 
    Module that splits an image into patches.
    We assumen square images and patches
    """

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        """
        img: (B, seq_len, C, H, W)
        Returns: (B, seq_len, num_patches, patch_dim)
        """
        B, seq_len, C, H, W = img.shape
        assert H % self.patch_size == 0, f"H={H} not divisible by patch_size={self.patch_size}"
        assert W % self.patch_size == 0, f"W={W} not divisible by patch_size={self.patch_size}"
        num_patch_H = H // self.patch_size
        num_patch_W = W // self.patch_size

        patch_data = img.reshape(
            B, seq_len, C, num_patch_H, self.patch_size, num_patch_W, self.patch_size
        )
        # permute to bring patch grid together
        # -> (B, seq_len, num_patch_H, num_patch_W, C, patch_size, patch_size)
        patch_data = patch_data.permute(0, 1, 3, 5, 2, 4, 6)
        num_patches = num_patch_H * num_patch_W
        patch_dim = C * self.patch_size * self.patch_size
    
        patch_data = patch_data.reshape(B, seq_len, num_patches, patch_dim) # -> (B, seq_len, num_patches, patch_dim)

        return patch_data
    
class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional encoding 

    Args:
    -----
    d_model: int
        Dimensionality of the slots/tokens
    max_len: int
        Length of the sequence.
    """

    def __init__(self, d_model, max_len=64):
        """
        Initializing the positional encoding
        """
        super().__init__()
        self.d_model = d_model #  The dimensionality of token embeddings
        self.max_len = max_len #  Maximum sequence length the model can handle (default 64)

        # initializing embedding
        self.pe = self._get_pe()
        return

    def _get_pe(self):
        """
        Initializing the temporal positional encoding given the encoding mode
        """
        max_len = self.max_len
        d_model = self.d_model
        
        pe = torch.zeros(max_len, d_model) # Creates a zero tensor - one row per position
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # Even dimensions get sine
        pe[:, 1::2] = torch.cos(position * div_term) # Odd dimensions get cosine
        pe = pe.view(1, max_len, d_model)
        return pe

    def forward(self, x):
        """
        Adding the positional encoding to the input tokens of the transformer
        """
        if x.device != self.pe.device:
            self.pe = self.pe.to(x.device)
        batch_size, seq_len, num_tokens, token_dim = x.shape
        # Repeat for batch and truncate to actual sequence length
        cur_pe = self.pe.repeat(batch_size, seq_len, 1, 1)[:, :, :num_tokens, :]
        # print(f"Cur pe shape: {cur_pe.shape}")
        y = x + cur_pe # Adding the positional encoding to the input tokens
        return y        
    