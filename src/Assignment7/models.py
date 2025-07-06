from utils import *
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
import random
import re
from transformations import *
from dataloader import KTHActionDataset
from torch.utils.tensorboard import SummaryWriter
import math


'''
Multi-Head Self-Attention module FLOW SUMMARY:

# Input: [32, 10, 17, 128]
# ↓ (split_into_heads)
# [1280, 17, 32] - for efficient attention computation
# ↓ (attention)
# [1280, 17, 32] - after attention
# ↓ (merge_heads)
# [320, 17, 128] - heads merged back
# ↓ (out_proj)
# [320, 17, 128] - after projection
# ↓ (reshape in forward)
# [32, 10, 17, 128] - back to original shape

'''
class MultiHeadSelfAttention(nn.Module):
    """ 
    Self-Attention module

    Args:
    -----
    token_dim: int
        Dimensionality of the tokens in the transformer
    inner_dim: int
        Dimensionality used for attention
    """

    def __init__(self, token_dim, attn_dim, num_heads):
        """ """
        super().__init__()
        self.token_dim = token_dim #  Embedding size per token, here called D. N number of tokens
        self.attn_dim = attn_dim # the dimension of the attention vector
        self.num_heads = num_heads 
        assert num_heads >= 1 # multi-head attention
        assert attn_dim % num_heads == 0, f"attn_dim = {attn_dim} must be divisible by num_heads = {num_heads}..."
        self.head_dim = attn_dim // num_heads

        # query, key and value projections
        self.q = nn.Linear(token_dim, attn_dim, bias=False) 
        self.k = nn.Linear(token_dim, attn_dim, bias=False) 
        self.v = nn.Linear(token_dim, attn_dim, bias=False) 

        # output projection
        self.out_proj = nn.Linear(attn_dim, token_dim, bias=False) # back to the original input dimension
        return
    
    def attention(self, query, key, value):
        """
        Computing self-attention

        All (q,k,v).shape ~ (batch_size * seq_len * self.num_heads, num_tokens, self.head_dim)
        """
        scale = (query.shape[-1]) ** (-0.5) # smoothing gradiants to work better with softmax

        # similarity between each query and the keys
        similarity = torch.bmm(query, key.permute(0, 2, 1)) * scale  # ~(B, N, N) batch-wise matrix multiplication, permmute here acts as traspose for dimentions matching
        attention = similarity.softmax(dim=-1) # softmax across each row 
        self.attention_map = attention # for visualization \latter

        # attention * values
        output = torch.bmm(attention, value)
        return output

    def split_into_heads(self, x):  # TODO: check if this is correct
        """
        Splitting a vector into multiple heads
        """
        # print(f"Input x shape: {x.shape}")

        batch_size, seq_len, num_tokens, token_dim = x.shape  # [32, 10, 17, 128]
        # print(f'number of heads: {self.num_heads}')
        # print(f'head dim: {self.head_dim}')
        # print(f"Input x shape: {x.shape}")
        
        # Reshape to combine batch and sequence dimensions for processing
        x = x.reshape(batch_size * seq_len, num_tokens, token_dim)  # [320, 17, 128]
        # print(f"Reshaped x shape: {x.shape}")
        
        # Split the token dimension into heads
        x = x.view(batch_size * seq_len, num_tokens, self.num_heads, self.head_dim)  # [320, 17, 4, 32]
        # print(f"After view x shape: {x.shape}")
        
        # Permute to get heads dimension first for independent attention
        x = x.permute(0, 2, 1, 3)  # [320, 4, 17, 32]
        # print(f"After permute x shape: {x.shape}")
        
        # Reshape to combine batch*seq and heads for batch processing
        y = x.reshape(batch_size * seq_len * self.num_heads, num_tokens, self.head_dim)  # [1280, 17, 32]
        # print(f"Final y shape: {y.shape}")
        
        return y

    def merge_heads(self, x):
        """
        Rearranging heads back to original head structure
        """
        _, num_tokens, dim_head = x.shape # [1280, 17, 32]
        y = x.reshape(-1, self.num_heads, num_tokens, dim_head) # --> [320, 4, 17, 32]
        y = y.reshape(-1, num_tokens, self.num_heads * dim_head) # --> [320, 17, 128]
        return y


    def forward(self, x):
        """ 
        Forward pass through Self-Attention module
        """
        # Store original shape to restore later
        original_shape = x.shape  # [32, 10, 17, 128]
        batch_size, seq_len, num_tokens, token_dim = original_shape
        
        # linear projections and splitting into heads:
        # (B, N, D) --> (B, N, Nh, Dh) --> (B * Nh, N, Dh)
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = self.split_into_heads(q) # [1280, 17, 32]
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        # applying attention equation
        vect = self.attention(query=q, key=k, value=v)
        # print(f"Vect shape: {vect.shape}")
        # rearranging heads: (B * Nh, N, Dh) --> (B*T, N, D)
        y = self.merge_heads(vect)  # [320, 17, 128]
        # print(f"Y SHAPE AFTER MERGE HEADS: {y.shape}")
        y = self.out_proj(y) #(B, N, token_dim) --> [320,17,128]
        # print(f"Y SHAPE AFTER OUT PROJ: {y.shape}")
        # Reshape back to original 4D shape
        y = y.reshape(batch_size, seq_len, num_tokens, token_dim)  # [32, 10, 17, 128]
        # print(f"Y SHAPE AFTER RESHAPE: {y.shape}")
        return y
    

class MLP(nn.Module):
    """
    2-Layer Multi-Layer Perceptron used in transformer blocks
    
    Args:
    -----
    in_dim: int
        Dimensionality of the input embeddings to the MLP
    hidden_dim: int
        Hidden dimensionality of the MLP
    """
    
    def __init__(self, in_dim, hidden_dim):
        """ MLP Initializer """
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),  ## NOTE: GELU activation function used in FCL for transformers!
                nn.Linear(hidden_dim, in_dim),
            )
        
    def forward(self, x):
        """ Forward """
        y = self.mlp(x)
        return y
    

class TransformerBlock(nn.Module):
    """
    Transformer block using self-attention

    Args:
    -----
    token_dim: int
        Dimensionality of the input tokens
    attn_dim: int
        Inner dimensionality of the attention module. Must be divisible be num_heads
    num_heads: int
        Number of heads in the self-attention mechanism
    mlp_size: int
        Hidden dimension of the MLP module
    """

    def __init__(self, token_dim, attn_dim, num_heads, mlp_size):
        """ Module initializer """
        super().__init__()
        self.token_dim = token_dim
        self.mlp_size = mlp_size
        self.attn_dim = attn_dim
        self.num_heads = num_heads

        # MHA
        self.ln_att = nn.LayerNorm(token_dim, eps=1e-6) # Layer normalization
        self.attn = MultiHeadSelfAttention(
                token_dim=token_dim,
                attn_dim=attn_dim,
                num_heads=num_heads
            ) # ---> [320,17,128]
        
        # MLP
        self.ln_mlp = nn.LayerNorm(token_dim, eps=1e-6) # Layer normalization
        self.mlp = MLP(
                in_dim=token_dim,
                hidden_dim=mlp_size,
            )
        return


    def forward(self, inputs):
        """
        Forward pass through transformer encoder block.
        We assume the more modern PreNorm design
        """
        # assert inputs.ndim == 3, f"Inputs to the transformer block must be of shape (B, N, D), but got {inputs.shape}"
        # print(f"INPUTS SHAPE: {inputs.shape}") 
 
        # Self-attention.
        x = self.ln_att(inputs)
        # print(f"X SHAPE BEFORE ATTENTION: {x.shape}")
        x = self.attn(x) # should return [32, 10, 17, 128]
        assert x.shape == inputs.shape, f"X shape: {x.shape} and inputs shape: {inputs.shape} MUST BE THE SAME (input and output of the attention block)"
        y = x + inputs # residual connection - both are now 4D [32, 10, 17, 128]

        # MLP
        z = self.ln_mlp(y)
        z = self.mlp(z)
        z = z + y # residual connection

        return z


    def get_attention_masks(self):
        """ Fetching last computer attention masks """
        attn_masks = self.attn.attention_map
        N = attn_masks.shape[-1]
        attn_masks = attn_masks.reshape(-1, self.num_heads, N, N)
        return attn_masks
    
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
    
class ViT(nn.Module):
    """ 
    Vision Transformer for image classification
    """

    def __init__(self, patch_size, token_dim, attn_dim, num_heads, mlp_size, num_tf_layers, num_classes, C, max_len=64):
        """ Model initializer
         num_tf_layers : The number of transformer blocks that we need
           """
        super().__init__()

        # breaking image into patches, and projection to transformer token dimension
        self.pathchifier = Patchifier(patch_size)

        ''' Creating the embedding for each image patch/token'''
        self.patch_projection = nn.Sequential(   
                nn.LayerNorm(patch_size * patch_size * C),
                nn.Linear(patch_size * patch_size * C, token_dim) # token_dim = token embedding
            )

        # adding CLS token and positional embedding
        '''nn.Parameter: To assign a tensor as Module attributes they are automatically added to the list of
        its parameters, and will appear e.g. in ~Module.parameters iterator. when requires_grad = True ---> the tensor
        is updated through training with GD. 
        / (token_dim ** 0.5): a common trick to stabilize training by keeping the scale of weights roughly controlled (similar to Xavier initialization)'''
        self.cls_token = nn.Parameter(torch.randn(1, token_dim) / (token_dim ** 0.5), requires_grad=True)
        self.pos_emb = PositionalEncoding(token_dim, max_len=max_len) # return token embeddings + positional encoding

        # cascade of transformer blocks
        transformer_blocks = [
            TransformerBlock(
                    token_dim=token_dim,
                    attn_dim=attn_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size
                )
            for _ in range(num_tf_layers)
        ]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        # classifier
        self.classifier = nn.Linear(token_dim, num_classes)
        return

    
    def forward(self, x): # full Transformer encoder block forward pass
        """ 
        Forward pass
        """
        B = x.shape[0]  # (B, 10, 1, 64, 64)
        seq_len = x.shape[1]
        C = x.shape[2]
        
        # breaking image into patches, and projection to transformer token dimension
        patches = self.pathchifier(x)  # ---> (B, 10, num_patches, patch_dim) e.g. [32, 10, 16, 256]
        patch_tokens = self.patch_projection(patches)  # ---> [32,10,16,128]
        # print(f"Patch tokens shape: {patch_tokens.shape}")
        # concatenating CLS token and adding positional embeddings
        cur_cls_token = self.cls_token.unsqueeze(0).repeat(B, 1, 1) # shape: (B, 1, D).i.e [32,1,128]  B copies of the token, one for each sample in the batch.
        # print(f"Cur cls token shape: {cur_cls_token.shape}")

        cur_cls_token = cur_cls_token.repeat(1, seq_len, 1).unsqueeze(2) # shape: (B, seq_len, 1, D) i.e [32,10,1,128]
        # print(f"New Cur cls token shape: {cur_cls_token.shape}")

        tokens = torch.cat([cur_cls_token, patch_tokens], dim=2)  # ~(B, 10, 1 + 16, D) So now, each input sequence starts with the [CLS] token, followed by the patch tokens.
        # ---> one token for all the patches of one image, not one per sequence. Summary token of all image
        # print(f"Tokens shape: {tokens.shape}") # [32, 10, 17, 128]

        tokens_with_pe = self.pos_emb(tokens) #tokens + positional encoding
        # print(f"Tokens with pe shape: {tokens_with_pe.shape}") # same shape : [32, 10, 17, 128]

        # processing with transformer
        out_tokens = self.transformer_blocks(tokens_with_pe)
        # print(f"Out tokens shape: {out_tokens.shape}")
        # Extract CLS token from each frame (position 0 in num_tokens dimension)
        out_cls_token = out_tokens[:, :, 0]  # Shape: [32, 10, 128]
        # print(f"Out cls token shape: {out_cls_token.shape}")
        
        # NOTE: For video classification, we'll use the mean of CLS tokens across all frames
        out_cls_token = out_cls_token.mean(dim=1)  # Shape: [32, 128]
        # print(f"Final cls token shape: {out_cls_token.shape}")

        # classification
        logits = self.classifier(out_cls_token)
        return logits


    def get_attn_mask(self):
        """
        Fetching the last attention maps from all TF Blocks
        """
        attn_masks = [tf.get_attention_masks() for tf in self.transformer_blocks]
        return attn_masks
