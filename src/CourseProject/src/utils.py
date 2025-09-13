import shutil
import os
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import math
import random
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    
    def set_random_seed(random_seed=None):
        """
        Using random seed for numpy and torch
        """
        if(random_seed is None):
            random_seed = 42
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        return
    
    def init_weights(self, m):
        # initialize nn.Linear and nn.LayerNorm
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            


def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f


def save_model(model, optimizer, epoch, stats, experiment_name):
    """ Saving model checkpoint """
    
    if(not os.path.exists("checkpoints")):
        os.makedirs("checkpoints")
    savepath = f"checkpoints/checkpoint_{experiment_name}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    return


def load_model(model, optimizer, savepath):
    """ Loading pretrained checkpoint """
    
    checkpoint = torch.load(savepath, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    
    return model, optimizer, epoch, stats


def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def visualize_attention(image, attention_maps, patch_size=16, img_size=64):
    """ Overlaying the attention maps on the image """
    num_layers = len(attention_maps)
    num_heads, num_tokens = attention_maps[0].shape
    patches_per_side = img_size // patch_size
    num_patches = patches_per_side * patches_per_side
    
    # first displaying raw image
    fig, ax = plt.subplots(1, num_layers + 1)
    fig.set_size_inches(30, 5)
    ax[0].imshow(image, cmap='gray')
    ax[0].axis("off")
    ax[0].set_title("Image", fontsize=20)

    # displaying attention from each layer
    image_unnorm = (image * 255).astype(np.uint8)
    H, W = image.shape[:2]
    for i in range(num_layers):
        cur_attn = attention_maps[i][:, 1:]  # current attn and removing [CLS] token

        attn = cur_attn.mean(axis=0)  # average across heads 
        attn = attn / attn.max()  # renormalization
        attn_grid = attn.reshape(patches_per_side, patches_per_side)  # mapping back to image

        # Resize to image resolution        
        attn_up = cv2.resize(attn_grid, (W, H), interpolation=cv2.INTER_CUBIC)
        # attn_up = cv2.resize(attn_grid, (W, H), interpolation=cv2.INTER_NEAREST)

        # cmap = "jet"
        cmap = "coolwarm"
        
        im = ax[i+1].imshow(image, cmap='gray')
        ax[i+1].imshow(attn_up, cmap='gray', alpha=0.01, extent=(0, W, H, 0))
        cbar = plt.colorbar(ax[i+1].imshow(attn_up, cmap=cmap, alpha=0.8, extent=(0, W, H, 0), vmin=0, vmax=1), ax=ax[i+1], fraction=0.046, pad=0.04, cmap=cmap)
        cbar.set_label('Attention Intensity', fontsize=15)
        ax[i+1].axis('off')
        ax[i+1].set_title(f"Attention Layer {i+1}/{num_layers}", fontsize=20)
        
    plt.show()

def visualize_progress(loss_iters, train_loss, val_loss, valid_acc, start=0):
    """ Visualizing loss and accuracy """
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(24,5)
    
    smooth_loss = smooth(loss_iters, 31)
    ax[0].plot(loss_iters, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_loss, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_title("Training Progress")
    
    epochs = np.arange(len(train_loss)) + 1
    ax[1].plot(epochs, train_loss, c="red", label="Train Loss", linewidth=3)
    ax[1].plot(epochs, val_loss, c="blue", label="Valid Loss", linewidth=3)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_title("Loss Curves")
    
    epochs = np.arange(len(val_loss)) + 1
    ax[2].plot(epochs, valid_acc, c="red", label="Valid accuracy", linewidth=3)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Accuracy (%)")
    ax[2].set_title(f"Valdiation Accuracy (max={round(np.max(valid_acc),2)}% @ epoch {np.argmax(valid_acc)+1})")
    
    plt.show()
    return


def visualize_masked_image(img, mask, patch_size=32):
    """
    Visualize original vs masked image.

    Args:
        img: [C, H, W] tensor (one image)
        mask: [N] tensor (0 = keep, 1 = masked)
        patch_size: size of each patch
    """
    C, H, W = img.shape
    num_patches_per_side = H // patch_size


    img = img.permute(1, 2, 0)/255.0

    # Make a copy for masked image
    masked_img = img.numpy().copy()

    # Apply mask by blacking out patches
    idx = 0
    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            if mask[idx] == 1:  # masked
                y0, y1 = i * patch_size, (i + 1) * patch_size
                x0, x1 = j * patch_size, (j + 1) * patch_size
                masked_img[y0:y1, x0:x1, :] = 0.0
            idx += 1

    # Plot side by side
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(masked_img)
    axs[1].set_title("Masked")
    axs[1].axis("off")

    plt.show()
