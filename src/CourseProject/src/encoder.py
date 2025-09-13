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
# from transformations import *
from utils import Patchifier,PositionalEncoding
from torch.utils.tensorboard import SummaryWriter
import math


class VitEncoder(nn.Module):
    """ 
    Vision Transformer for image reconstruction task
    """
    def __init__(self, 
                 patch_size, 
                 embed_dim, 
                 attn_dim, 
                 num_heads, 
                 mlp_size, 
                 encoder_depth, 
                 in_chans = 3, 
                 max_len  = 64,
                 mask_ratio = 0.75,
                 norm_pix_loss=False):
        
        """ Model initializer
         encoder_depth : The number of transformer blocks that we need in encoder
         max_len : Maximum sequence length (number of emdeddings) the model can handle
         norm_pix_loss : if we need to normalize the loss for pixels inside a patch
           """
        super().__init__()
        
        self.initialize_weights()
        self.mask_ratio = mask_ratio
        # breaking image into patches, and projection to transformer token dimension
        self.pathchifier = Patchifier(patch_size)

        ''' Creating the embedding for each image patch/token'''
        self.patch_projection = nn.Sequential(   
                nn.LayerNorm(patch_size * patch_size * in_chans),
                nn.Linear(patch_size * patch_size * in_chans, embed_dim) # embed_dim = token embedding
            )

        self.pos_emb = PositionalEncoding(embed_dim, max_len = max_len) # return token embeddings + positional encoding

        # cascade of transformer blocks
        transformer_blocks = [
            TransformerBlock(
                    embed_dim = embed_dim,
                    attn_dim  = attn_dim,
                    num_heads = num_heads,
                    mlp_size  = mlp_size
                )
            for _ in range(encoder_depth)
        ]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        self.norm = nn.LayerNorm(embed_dim)
        self.norm_pix_loss = norm_pix_loss

        return

    def initialize_weights(self):
        ''' TODO '''
        pass

    def random_masking(self, x):
        """
        Perform random masking on patchified sequences.
        
        Args:
            x: [B, T, N, D] 
                - B = batch
                - T = sequence length
                - N = number of patches
                - D = patch embedding dimension (flattened patch)
            mask_ratio: float, fraction of patches to mask

        Returns:
            x_masked: [B, T, N_keep, D]   (only visible patches)
            mask:     [B, T, N]           (0 = keep, 1 = masked)
            ids_restore: [B, T, N]        (indices to restore order)
        
        Code initiallly borrowed from https://github.com/facebookresearch/mae/blob/main/models_mae.py and adjusted to our task
        """
        
        B, T, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio)) # How many tokens/patches to keep?

        # Generate random noise per sequence element
        noise = torch.rand(B, T, N, device=x.device)

        # Sort patches by noise
        ids_shuffle = torch.argsort(noise, dim=-1)          # [B, T, N]
        ids_restore = torch.argsort(ids_shuffle, dim=-1)    # [B, T, N]

        # Keep the first len_keep patches
        ids_keep = ids_shuffle[:, :, :len_keep]             # [B, T, N_keep]
        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, -1, D)

        # Select the kept patches
        x_masked = torch.gather(x, dim=2, index=ids_keep_expanded)  # [B, T, N_keep, D]

        # Build mask: 0 = keep, 1 = masked
        mask = torch.ones([B, T, N], device=x.device)
        mask[:, :, :len_keep] = 0
        mask = torch.gather(mask, dim=2, index=ids_restore)  # reorder to original
        '''
        x_masked: only the unmasked (kept) tokens.
        mask: binary mask for the original sequence (0=keep, 1=mask). id's for which tokens to keep and mask
        ids_restore: the permutation needed to restore original order.
        '''
        return x_masked, mask, ids_restore

    def forward(self, x): # full Transformer encoder block forward pass
        """ 
        Forward pass
        """
        B = x.shape[0]  
        C = x.shape[2]
        
        # breaking image into patches, and projection to transformer token dimension
        patches = self.pathchifier(x)  # ---> (B, 10, num_patches, patch_dim)
        patch_tokens = self.patch_projection(patches)  
        # print(f"Patch tokens shape: {patch_tokens.shape}")

        tokens_with_pe = self.pos_emb(patch_tokens) #tokens + positional encoding
        # print(f"Tokens with pe shape: {tokens_with_pe.shape}")


        ''' masking tokens'''
        x, mask, ids_restore = self.random_masking(tokens_with_pe) 

        # apply Transformer blocks
        x = self.transformer_blocks(x)
        # print(f"Out tf_block tokens shape: {x.shape}")

        x = self.norm(x)

        return x, mask, ids_restore


    def get_attn_mask(self):
        """
        Fetching the last attention maps from all TF Blocks
        """
        attn_masks = [tf.get_attention_masks() for tf in self.transformer_blocks]
        return attn_masks



class MultiHeadSelfAttention(nn.Module):
    """ 
    Self-Attention module

    Args:
    -----
    embed_dim: int
        Dimensionality of the tokens in the transformer
    inner_dim: int
        Dimensionality used for attention
    """

    def __init__(self, embed_dim, attn_dim, num_heads):
        """ """
        super().__init__()
        self.embed_dim = embed_dim #  Embedding size per token, here called D. N number of tokens
        self.attn_dim = attn_dim # the dimension of the attention vector
        self.num_heads = num_heads 
        assert num_heads >= 1 # multi-head attention
        assert attn_dim % num_heads == 0, f"attn_dim = {attn_dim} must be divisible by num_heads = {num_heads}..."
        self.head_dim = attn_dim // num_heads

        # query, key and value projections
        self.q = nn.Linear(embed_dim, attn_dim, bias=False) 
        self.k = nn.Linear(embed_dim, attn_dim, bias=False) 
        self.v = nn.Linear(embed_dim, attn_dim, bias=False) 

        # output projection
        self.out_proj = nn.Linear(attn_dim, embed_dim, bias=False) # back to the original input dimension
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

        batch_size, seq_len, num_tokens, embed_dim = x.shape 
        # print(f'number of heads: {self.num_heads}')
        # print(f'head dim: {self.head_dim}')
        # print(f"Input x shape: {x.shape}")
        
        # Reshape to combine batch and sequence dimensions for processing
        x = x.reshape(batch_size * seq_len, num_tokens, embed_dim)  
        # print(f"Reshaped x shape: {x.shape}")
        
        # Split the token dimension into heads
        x = x.view(batch_size * seq_len, num_tokens, self.num_heads, self.head_dim)  
        # print(f"After view x shape: {x.shape}")
        
        # Permute to get heads dimension first for independent attention
        x = x.permute(0, 2, 1, 3) 
        # print(f"After permute x shape: {x.shape}")
        
        # Reshape to combine batch*seq and heads for batch processing
        y = x.reshape(batch_size * seq_len * self.num_heads, num_tokens, self.head_dim)  
        # print(f"Final y shape: {y.shape}")
        
        return y

    def merge_heads(self, x):
        """
        Rearranging heads back to original head structure
        """
        _, num_tokens, dim_head = x.shape 
        y = x.reshape(-1, self.num_heads, num_tokens, dim_head) 
        y = y.reshape(-1, num_tokens, self.num_heads * dim_head) 
        return y


    def forward(self, x):
        """ 
        Forward pass through Self-Attention module
        """
        # Store original shape to restore later
        original_shape = x.shape 
        batch_size, seq_len, num_tokens, embed_dim = original_shape
        
        # linear projections and splitting into heads:
        # (B, N, D) --> (B, N, Nh, Dh) --> (B * Nh, N, Dh)
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = self.split_into_heads(q) 
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        # applying attention equation
        vect = self.attention(query=q, key=k, value=v)
        # print(f"Vect shape: {vect.shape}")
        # rearranging heads: (B * Nh, N, Dh) --> (B*T, N, D)
        y = self.merge_heads(vect)  
        # print(f"Y SHAPE AFTER MERGE HEADS: {y.shape}")
        y = self.out_proj(y) #(B, N, embed_dim)
        # print(f"Y SHAPE AFTER OUT PROJ: {y.shape}")
        # Reshape back to original 4D shape
        y = y.reshape(batch_size, seq_len, num_tokens, embed_dim)  
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
    embed_dim: int
        Dimensionality of the input tokens
    attn_dim: int
        Inner dimensionality of the attention module. Must be divisible be num_heads
    num_heads: int
        Number of heads in the self-attention mechanism
    mlp_size: int
        Hidden dimension of the MLP module
    """

    def __init__(self, embed_dim, attn_dim, num_heads, mlp_size):
        """ Module initializer """
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_size = mlp_size
        self.attn_dim = attn_dim
        self.num_heads = num_heads

        # MHA
        self.ln_att = nn.LayerNorm(embed_dim, eps=1e-6) # Layer normalization
        self.attn = MultiHeadSelfAttention(
                embed_dim=embed_dim,
                attn_dim=attn_dim,
                num_heads=num_heads
            ) # ---> [320,17,128]
        
        # MLP
        self.ln_mlp = nn.LayerNorm(embed_dim, eps=1e-6) # Layer normalization
        self.mlp = MLP(
                in_dim=embed_dim,
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
        x = self.attn(x) 
        assert x.shape == inputs.shape, f"X shape: {x.shape} and inputs shape: {inputs.shape} MUST BE THE SAME (input and output of the attention block)"
        y = x + inputs # residual connection - both are now 4D 

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
