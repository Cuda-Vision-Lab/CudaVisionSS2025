from CourseProject.src.utiils.utils import *
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
from CourseProject.src.utiils.utils import Patchifier,PositionalEncoding
from torch.utils.tensorboard import SummaryWriter
import math
# from model_base import RandomMaskingMixin


class MaskEncoder(nn.Module):
    """
    Encodes segmentation masks into patch embeddings.
    
    Args:
        patch_size: int, size of patches for mask patching
        embed_dim: int, embedding dimension
        in_chans: int, number of input channels (1 for grayscale masks)
    """
    
    def __init__(self, patch_size, embed_dim, in_chans=1):
        super().__init__()

        self.mask_patchifier = Patchifier(patch_size)
        
        # Projection for mask patches
        self.mask_projection = nn.Sequential(
            nn.LayerNorm(patch_size * patch_size * in_chans),
            nn.Linear(patch_size * patch_size * in_chans, embed_dim)
        )
        
    def forward(self, masks):
        """
        Args:
            masks: [B, T, H, W] - segmentation masks
        
        Returns:
            mask_embeddings: [B, T, num_patches, embed_dim]
        """
        B, T, H, W = masks.shape
        
        # Convert to float32 to match LayerNorm expectations
        masks = masks.float()
        
        # Add channel dimension 
        if masks.dim() == 4:
            masks = masks.unsqueeze(2)  # [B, T, 1, H, W]
        
        # Patchify masks
        mask_patches = self.mask_patchifier(masks)  # [B, T, num_patches, patch_dim]
        
        # Project to embedding space
        mask_embeddings = self.mask_projection(mask_patches)  # [B, T, num_patches, embed_dim]
        
        return mask_embeddings


class BBoxEncoder(nn.Module):
    """
    Encodes bounding boxes into embeddings that can be used with the transformer.
    
    Args:
        embed_dim: int, embedding dimension for the transformer
        max_objects: int, maximum number of objects per frame
        bbox_dim: int, dimension of bbox coordinates (usually 4 for x1,y1,x2,y2)
    """
    
    def __init__(self, embed_dim, max_objects=11):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Learnable embeddings for bbox coordinates
        self.bbox_projection = nn.Sequential(
            nn.Linear(4, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Positional encoding for bbox order
        self.bbox_pos_encoding = nn.Parameter(torch.randn(max_objects, embed_dim // 4))
        
        # Final projection to combine all bbox features
        self.bbox_final_proj = nn.Linear(embed_dim + embed_dim // 4, embed_dim)
        
    def forward(self, bboxes):
        """
        Args:
            bboxes: [B, T, max_objects, bbox_dim] - bounding box coordinates
        
        Returns:
            bbox_embeddings: [B, T, max_objects, embed_dim]
        """
        B, T, max_objects, bbox_dim = bboxes.shape
        
        # Convert to float32 to match Linear layer expectations
        # bboxes = bboxes.float()
        
        # Project bbox coordinates to embedding space
        bbox_coords = bboxes.view(B * T * max_objects, bbox_dim)
        bbox_emb = self.bbox_projection(bbox_coords)  # [B*T*max_objects, embed_dim]
        bbox_emb = bbox_emb.view(B, T, max_objects, self.embed_dim)
        bbox_embeddings = bbox_emb
          
        return bbox_embeddings

class MultiModalVitEncoder(nn.Module):
# class MultiModalVitEncoder(RandomMaskingMixin, nn.Module):
    
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
                 max_objects=11,
                 use_masks=False,
                 use_bboxes=False,
                 norm_pix_loss=False):
        
        """ Model initializer
         encoder_depth : The number of transformer blocks that we need in encoder
         max_len : Maximum sequence length (number of emdeddings) the model can handle
         norm_pix_loss : if we need to normalize the loss for pixels inside a patch
           """
        super().__init__()
        
        # self.initialize_weights()
        self.embed_dim = embed_dim
        self.use_masks = use_masks
        self.use_bboxes = use_bboxes
        self.mask_ratio = mask_ratio
        
        # breaking image into patches, and projection to transformer token dimension
        self.patchifier = Patchifier(patch_size)

        ''' Image processing. Creating the embedding for each image patch/token'''
        self.patch_projection = nn.Sequential(   
                nn.LayerNorm(patch_size * patch_size * in_chans),
                nn.Linear(patch_size * patch_size * in_chans, embed_dim) # embed_dim = token embedding
            )

        # Mask processing
        if use_masks:
            self.mask_encoder = MaskEncoder(patch_size, embed_dim, in_chans=1)
        
        # Bounding box processing
        if use_bboxes:
            self.bbox_encoder = BBoxEncoder(embed_dim, max_objects)
        
        # Multi-modal embeddings
        if use_bboxes or use_masks:
            self.modality_embeddings = nn.Embedding(3, embed_dim)  # “tag” the input stream ---> 0: image, 1: mask, 2: bbox
        
        # Positional encoding
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
        
        Code initiallly adapted from https://github.com/facebookresearch/mae/blob/main/models_mae.py and adjusted to our task
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
        # return super().random_masking(x)

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    
    def forward(self, images, masks=None, bboxes=None): # full Transformer encoder block forward pass
        """ 
        Forward pass
        """
        B, T = images.shape[:2]  
        all_tokens = []
        all_masks = {}
        all_ids_restore = {}
        
        # Process images
        image_patches = self.patchifier(images)
        image_tokens = self.patch_projection(image_patches)
        image_tokens = self.pos_emb(image_tokens)

        # If using multiple modalities, add modality embedding
        if self.use_masks or self.use_bboxes:
            modality_emb = self.modality_embeddings(torch.zeros(B, T, 1, device=images.device, dtype=torch.long))
            image_tokens = image_tokens + modality_emb
        
        # Apply masking to images
        img_masked, img_mask, img_ids_restore = self.random_masking(image_tokens)
        all_tokens.append(img_masked)
        all_masks["image"] = img_mask
        all_ids_restore["image"] = img_ids_restore
        
        '''Single-modal input, only images'''
        if not self.use_masks and not self.use_bboxes:
            encoded_features = self.transformer_blocks(img_masked)
            encoded_features = self.norm(encoded_features)
            return encoded_features, all_masks, all_ids_restore
        
        '''Multi-modal processing'''
        # Process masks if provided
        if self.use_masks and masks is not None:
            mask_tokens = self.mask_encoder(masks['masks'])
            mask_tokens = self.pos_emb(mask_tokens)
            
            # Add modality embedding for masks
            modality_emb = self.modality_embeddings(torch.ones(B, T, 1, device=masks['masks'].device, dtype=torch.long))
            mask_tokens = mask_tokens + modality_emb
            
            # Apply masking to masks
            mask_masked, mask_mask, mask_ids_restore = self.random_masking(mask_tokens)
            all_tokens.append(mask_masked)
            all_masks["mask"] = mask_mask
            all_ids_restore["mask"] = mask_ids_restore
        
        # Process bounding boxes if provided
        if self.use_bboxes and bboxes is not None:
            bbox_tokens = self.bbox_encoder(bboxes)
            bbox_tokens = self.pos_emb(bbox_tokens)
            
            # Add modality embedding for bboxes
            modality_emb = self.modality_embeddings(torch.full((B, T, 1), 2, device=bboxes.device, dtype=torch.long))
            bbox_tokens = bbox_tokens + modality_emb
            
            # Apply masking to bboxes
            bbox_masked, bbox_mask, bbox_ids_restore = self.random_masking(bbox_tokens)
            all_tokens.append(bbox_masked)
            all_masks["bbox"] = bbox_mask
            all_ids_restore["bbox"] = bbox_ids_restore
        
        # Concatenate all tokens
        if len(all_tokens) > 1:
            # Pad shorter sequences to match the longest
            max_length = max(token.shape[2] for token in all_tokens)
            padded_tokens = []
            
            for token in all_tokens:
                if token.shape[2] < max_length:
                    # Pad with zeros
                    padding = torch.zeros(B, T, max_length - token.shape[2], self.embed_dim, 
                                        device=token.device, dtype=token.dtype)
                    token = torch.cat([token, padding], dim=2)
                padded_tokens.append(token)
            
            combined_tokens = torch.cat(padded_tokens, dim=2)
        else:
            combined_tokens = all_tokens[0]
        
           
        # apply Transformer blocks
        encoded_features = self.transformer_blocks(combined_tokens)
        # print(f"Out tf_block tokens shape: {x.shape}")

        encoded_features = self.norm(encoded_features)

        return encoded_features, all_masks, all_ids_restore


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
